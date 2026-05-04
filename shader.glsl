#version 430 core
#extension GL_ARB_shading_language_420pack : require
in vec2 fragCoord;
out vec4 FragColor;

uniform float iTime;
uniform vec2 iResolution;

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3 emissive;
    vec3 albedo;
    uint reflects;
    float roughness;
    float metallic;
};

struct HitResult {
    bool hit;
    float t;
    vec3 position;
    vec3 normal;
    Material material;
};

struct Triangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    Material material;
};

layout(std430, binding = 0) buffer Triangles {
    Triangle tris[];
};
uniform int numTriangles;
uniform int iSamples;
uniform vec3 iBackground;

// Hash function to generate pseudo-random value from a seed
float hash(float seed) {
    return fract(sin(seed) * 43758.5453123);
}

// 2D hash function
float hash2D(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 3D hash function for (x, y, time)
float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

// PCG hash — integer-based, no sin precision issues
uint rngState;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand() {
    rngState = pcg(rngState);
    return float(rngState) / 4294967296.0;
}

HitResult rayTriangleIntersection(Ray ray, vec3 v0, vec3 v1, vec3 v2, Material material) {
    HitResult result = HitResult(false, -1.0, vec3(0.0), vec3(0.0), material);

    vec3 rayOrigin = ray.origin;
    vec3 rayDirection = ray.direction;

    float epsilon = 0.0001;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(rayDirection, edge2);
    float a = dot(edge1, h);
    // Backface culling: a > 0 for front-face hits, a <= 0 for back-face / parallel.
    if (a < epsilon)
        return result;
    float f = 1.0 / a;
    vec3 s = rayOrigin - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return result;
    vec3 q = cross(s, edge1);
    float v = f * dot(rayDirection, q);
    if (v < 0.0 || u + v > 1.0)
        return result;
    float t = f * dot(edge2, q);
    if (t > 0.001) {
        result = HitResult(true, t, rayOrigin + rayDirection * t, normalize(cross(edge1, edge2)), material);
        return result; // Intersection
    } else
        return result; // Line intersection but not a ray intersection
}

HitResult raySceneIntersection(Ray ray) {
    HitResult closestHit = HitResult(
        false,
        1e30,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), vec3(0.0), 0u, 1.0, 0.0)
    );

    for (int i = 0; i < numTriangles; i++) {
        Triangle tri = tris[i];

        HitResult hit = rayTriangleIntersection(
            ray,
            tri.v0, tri.v1, tri.v2,
            tri.material
        );

        if (hit.hit && hit.t < closestHit.t) {
            closestHit = hit;
        }
    }

    return closestHit;
}

// Cosine-weighted hemisphere sample. Pairs with throughput *= albedo
// for a correct diffuse estimator (the cos and pdf cancel).
vec3 randomUnitHemisphere(vec3 normal) {
    float r1 = rand();
    float r2 = rand();
    float phi = 6.28318530718 * r1;
    float sinTheta = sqrt(r2);
    float cosTheta = sqrt(1.0 - r2);

    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * (sinTheta * cos(phi))
         + bitangent * (sinTheta * sin(phi))
         + normal * cosTheta;
}

// Trowbridge-Reitz GGX importance sample — returns a microfacet half-vector
// distributed by D(h). Roughness=0 collapses to H = N (perfect mirror).
vec3 sampleGGXHalf(vec3 normal, float roughness) {
    float r1 = rand();
    float r2 = rand();
    float a = roughness * roughness;
    float phi = 6.28318530718 * r1;
    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 H_local = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return tangent * H_local.x + bitangent * H_local.y + normal * H_local.z;
}

// Smith geometry (Schlick-GGX, direct-lighting variant)
float smithG(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    return gV * gL;
}

// Cook-Torrance PBR bounce: probabilistically pick specular GGX or Lambert diffuse.
// `throughputMult` is the BRDF*cos/pdf factor to multiply into the running throughput.
Ray pbrBounce(vec3 V, vec3 N, vec3 hitPos, Material mat, out vec3 throughputMult) {
    vec3 F0 = mix(vec3(0.04), mat.albedo, mat.metallic);
    float specProb = clamp(0.5 + 0.5 * mat.metallic, 0.25, 0.9);
    vec3 origin = hitPos + N * 1e-4;

    if (rand() < specProb) {
        vec3 H = sampleGGXHalf(N, mat.roughness);
        vec3 L = reflect(-V, H);
        if (dot(L, N) <= 0.0) {
            throughputMult = vec3(0.0);
            return Ray(origin, L);
        }
        float NdotV = max(dot(N, V), 1e-4);
        float NdotL = max(dot(N, L), 1e-4);
        float NdotH = max(dot(N, H), 1e-4);
        float VdotH = max(dot(V, H), 1e-4);

        vec3 F = F0 + (1.0 - F0) * pow(max(0.0, 1.0 - VdotH), 5.0);
        float G = smithG(NdotV, NdotL, mat.roughness);
        // BRDF * NdotL / pdf (GGX importance-sampled half-vector) = F * G * VdotH / (NdotH * NdotV)
        throughputMult = F * G * VdotH / (NdotH * NdotV * specProb);
        return Ray(origin, L);
    } else {
        vec3 L = randomUnitHemisphere(N);  // cosine-weighted Lambert
        vec3 kd = (1.0 - mat.metallic) * mat.albedo;
        throughputMult = kd / (1.0 - specProb);
        return Ray(origin, L);
    }
}

void handleHit(inout vec3 col, Ray ray, HitResult hit) {
    const int MAX_BOUNCES = 4;

    Ray currentRay = ray;
    HitResult currentHit = hit;

    vec3 throughput = vec3(1.0);
    vec3 emissive = vec3(0.0);

    for (int i = 0; i < MAX_BOUNCES; i++) {
        emissive += throughput * currentHit.material.emissive;

        if (currentHit.material.reflects == 0u)
            break;

        vec3 V = -currentRay.direction;
        vec3 throughputMult;
        Ray bouncedRay = pbrBounce(V, currentHit.normal, currentHit.position,
                                   currentHit.material, throughputMult);
        throughput *= throughputMult;

        if (max(max(throughput.x, throughput.y), throughput.z) <= 0.0)
            break;

        HitResult nextHit = raySceneIntersection(bouncedRay);
        if (!nextHit.hit) {
            emissive += throughput * iBackground;
            break;
        }

        currentRay = bouncedRay;
        currentHit = nextHit;
    }
    col = emissive;
}

void main() {
    rngState = uint(gl_FragCoord.x) * 1973u
             + uint(gl_FragCoord.y) * 9277u
             + floatBitsToUint(iTime) * 26699u;
    int SAMPLES = iSamples;

    vec2 uv = (gl_FragCoord.xy / iResolution) * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y; 

    // Simple pinhole camera
    vec3 camPos  = vec3(iTime, iTime - 5, -5.0);   // back and slightly above
    vec3 camTarget = vec3(0.0, 2.5, 0.0);  // look at center-ish
    vec3 camDir = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(camDir, vec3(0.0, 1.0, 0.0)));
    vec3 camUp = cross(camRight, camDir);

    float fov = radians(45.0);
    float z = 1.0 / tan(fov * 0.5);

    vec2 pixelSize = vec2(2.0 / iResolution.y);

    vec3 accumColor = vec3(0.0);
    for (int i = 0; i < SAMPLES; i++) {
        vec2 jitter = (vec2(rand(), rand()) - 0.5) * pixelSize;
        vec2 sampleUv = uv + jitter;
        vec3 rayDir = normalize(sampleUv.x * camRight + sampleUv.y * camUp + z * camDir);
        Ray ray = Ray(camPos, rayDir);

        HitResult result = raySceneIntersection(ray);

        vec3 col;
        if (result.hit) {
            handleHit(col, ray, result);
        } else {
            col = iBackground;
        }
        accumColor += col;
    }
    vec3 col = accumColor / float(SAMPLES);

    FragColor = vec4(col, 1.0);
}