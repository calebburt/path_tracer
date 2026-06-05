#version 430 core
#extension GL_ARB_shading_language_420pack : require
in vec2 fragCoord;
out vec4 FragColor;

uniform float iTime;
uniform vec2 iResolution;
uniform float iApertureSize;
uniform int numTriangles;
uniform int iSamples;
uniform vec3 iBackground;

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

// Hash function to generate pseudo-random value from a seed
float hash(float seed) {
    return fract(sin(seed) * 43758.5453123);
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

    float fogT; // Fog can be anywhere, randomly choose a point normal-distributed, centered on 0 and with a standard deviation of 5 units, so that most fog hits are within ~10 units but some can be farther.
    {
        float u1 = rand();
        float u2 = rand();
        float z = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2); // Box-Muller transform
        fogT = abs(z * 10.0); // Scale by standard deviation and take absolute value to get a positive distance
    }

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

    if (fogT < closestHit.t) {
        closestHit = HitResult(
            true,
            fogT,
            ray.origin + ray.direction * fogT,
            vec3(rand(), rand(), rand()),
            Material(vec3(0), vec3(1), 0u, 1.0, 0.0)
        );
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

// PBR bounce: probabilistically pick specular or diffuse using Fresnel weighting.
// `throughputMult` is the color to multiply into the running throughput.
Ray pbrBounce(vec3 V, vec3 N, vec3 hitPos, Material mat, out vec3 throughputMult) {
    vec3 F0 = mix(vec3(0.04), mat.albedo, mat.metallic);
    vec3 origin = hitPos + N * 1e-4;

    // Fresnel-Schlick: F(θ) = F0 + (1 - F0)(1 - cos(θ))^5
    float cosTheta = clamp(dot(V, N), 0.0, 1.0);
    float f = pow(1.0 - cosTheta, 5.0);
    vec3 fresnel = F0 + (vec3(1.0) - F0) * f;

    // Use luminance of Fresnel as specular probability (0.04 for dielectrics at normal, 1.0 for metals)
    float specProb = dot(fresnel, vec3(0.299, 0.587, 0.114));

    if (rand() < specProb) {
        // Specular reflection
        vec3 L = reflect(-V, N);
        vec3 randomDir = randomUnitHemisphere(N);
        L = mix(L, randomDir, mat.roughness); // roughness = 0 is perfect mirror, 1 is fully random
        L = normalize(L);

        // Dielectrics reflect white, metals reflect their albedo, both modulated by Fresnel
        vec3 specColor = mix(vec3(1.0), mat.albedo, mat.metallic);
        throughputMult = fresnel * specColor / specProb;
        return Ray(origin, L);
    } else {
        // Diffuse reflection (only non-metals scatter diffusely)
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

        vec3 V = -currentRay.direction;
        vec3 throughputMult;
        Ray bouncedRay = pbrBounce(V, currentHit.normal, currentHit.position,
                                   currentHit.material, throughputMult);
        throughput *= throughputMult;

        // Russian roulette: probabilistically terminate low-throughput paths
        float survivalProb = min(1.0, dot(throughput, vec3(0.299, 0.587, 0.114))); // Match how perceptible to humans the color is
        if (rand() > survivalProb)
            break;
        throughput /= survivalProb;

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

    // Length where the ray hits the focal plane, controls depth of field strength.
    // This does NOT control the field of view — FOV is handled by 'z'.
    // focalLength is the distance from the camera where rays should converge to create a sharp image.
    float focalLength = distance(camPos, camTarget);

    vec2 pixelSize = vec2(2.0 / iResolution.y);

    vec3 accumColor = vec3(0.0);
    for (int i = 0; i < SAMPLES; i++) {
        vec2 jitter = (vec2(rand(), rand()) - 0.5) * pixelSize;
        vec2 aperturePoint = (vec2(rand(), rand()) - 0.5) * iApertureSize;
        vec3 camPosJittered = camPos + camRight * aperturePoint.x + camUp * aperturePoint.y;
        vec2 sampleUv = uv + jitter;
        // Primary ray direction based on pinhole projection (controls FOV)
        vec3 rayDir = normalize(sampleUv.x * camRight + sampleUv.y * camUp + z * camDir);
        // If using depth of field, shift origin across the aperture and reaim the ray so that
        // it passes through the focal plane at distance 'focalLength' from the original camera position.
        // This makes focalLength determine the plane of perfect focus without altering FOV.
        vec3 focalPoint = camPos + rayDir * focalLength;
        rayDir = normalize(focalPoint - camPosJittered);
        Ray ray = Ray(camPosJittered, rayDir);

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
