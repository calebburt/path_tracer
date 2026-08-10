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
    float ior;
    vec3 albedo;
    uint dielectric;
    float alpha;
    float roughness;
    float metallic;
};

struct HitResult {
    bool hit;
    bool frontFace;
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

struct RNG {
    uvec2 state;
    uvec2 inc;
};

RNG rng;

uint pcg32(inout RNG rng) {
    // Simplified 32-bit LCG-based RNG
    // Use state.x as the 32-bit state.
    uint s = rng.state.x;
    s = s * 1664525u + 1013904223u; // LCG step
    // xorshift32 mix
    uint x = s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng.state.x = s;
    return x;
}

float rand() {
    return float(pcg32(rng)) * (1.0 / 4294967296.0);
}

HitResult rayTriangleIntersection(Ray ray, vec3 v0, vec3 v1, vec3 v2, Material material) {
    HitResult result = HitResult(false, true, -1.0, vec3(0.0), vec3(0.0), material);

    vec3 rayOrigin = ray.origin;
    vec3 rayDirection = ray.direction;

    float epsilon = 0.0001;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(rayDirection, edge2);
    float a = dot(edge1, h);
    // Backface detection: a > 0 for front-face hits, a <= 0 for back-face / parallel.
    if (a < epsilon)
        result.frontFace = false;
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
        vec3 normal = cross(edge1, edge2);
        if (length(normal) < epsilon)
            return result;
        result = HitResult(true, result.frontFace, t, rayOrigin + rayDirection * t, normalize(normal), material);
        return result; // Intersection
    } else
        return result; // Line intersection but not a ray intersection
}

HitResult raySceneIntersection(Ray ray, bool cullBackface) {
    HitResult closestHit = HitResult(
        false,
        true,
        1e30,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), 1.0, vec3(0.0), 0u, 1.0, 1.0, 0.0)
    );

    for (int i = 0; i < numTriangles; i++) {
        Triangle tri = tris[i];

        HitResult hit = rayTriangleIntersection(
            ray,
            tri.v0, tri.v1, tri.v2,
            tri.material
        );

        if (cullBackface && !hit.frontFace)
            hit.hit = false;

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

vec3 randomUnitSphere()
{
    while (true)
    {
        vec3 p = vec3(
            rand() * 2.0 - 1.0,
            rand() * 2.0 - 1.0,
            rand() * 2.0 - 1.0
        );

        float len2 = dot(p, p);

        if (len2 > 0.0 && len2 < 1.0)
            return p * inversesqrt(len2);
    }
}

// PBR bounce: probabilistically pick specular or diffuse using Fresnel weighting.
// `throughputMult` is the color to multiply into the running throughput.
Ray pbrBounce(vec3 V, vec3 N, HitResult hit, Material mat, out vec3 throughputMult) {
    const float EPS=1e-4;
    vec3 origin=hit.position+N*EPS;

    if(bool(mat.dielectric)) {
        float eta=mat.ior;
        float etaRatio=hit.frontFace?(1.0/eta):eta;

        float cosTheta=min(dot(-V,N),1.0);
        float sinTheta=sqrt(max(0.0,1.0-cosTheta*cosTheta));

        float r0=(eta-1.0)/(eta+1.0);
        r0*=r0;

        float fresnel=r0+(1.0-r0)*pow(1.0-cosTheta,5.0);
        bool tir=etaRatio*sinTheta>1.0;

        vec3 dir;

        if(tir||rand()>fresnel) {
            dir=reflect(-V,N);

            if(mat.roughness>0.0)
                dir=normalize(dir+randomUnitSphere()*mat.roughness);

            throughputMult=vec3(1.0);
        } else {
            dir=refract(-V,N,etaRatio);

            if (mat.roughness>0.0)
                dir=normalize(dir+randomUnitSphere()*mat.roughness);

            throughputMult=mat.albedo;
        }

        return Ray(hit.position+dir*EPS,dir);
    }

    vec3 F0=mix(vec3(0.04), mat.albedo,mat.metallic);

    float NdotV=max(dot(N, V), 0.0);

    vec3 fresnel=F0+(vec3(1.0)-F0) * pow(1.0-NdotV,5.0);

    float specProb=clamp(
        dot(fresnel,vec3(0.2126,0.7152,0.0722)),
        0.05,
        0.95
    );

    if(rand() < specProb) {
        vec3 dir=normalize(mix(
            reflect(-V,N),
            randomUnitHemisphere(N),
            mat.roughness*mat.roughness
        ));

        throughputMult=
            mix(vec3(1.0),mat.albedo,mat.metallic)*
            fresnel/specProb;

        return Ray(origin,dir);
    }

    vec3 dir=randomUnitHemisphere(N);
    vec3 kd=mat.albedo*(1.0-mat.metallic)*(vec3(1.0)-fresnel);

    throughputMult=kd/(1.0-specProb);

    return Ray(origin,dir);
}

void handleHit(inout vec3 col, Ray ray, HitResult hit) {
    const int MAX_BOUNCES = 10;
    const float epsilon = 0.0001;

    Ray currentRay = ray;
    vec3 throughput = vec3(1.0);
    vec3 accum = vec3(0.0);

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        // Find next surface intersection for this ray
        HitResult nextHit = raySceneIntersection(currentRay, true);
        float distToSurface = nextHit.hit ? nextHit.t : 1e30;

        if (nextHit.material.alpha < 1.0 - epsilon && !bool(nextHit.material.dielectric)) {

            // Fog density: alpha = 0 -> no medium, alpha = 1 -> dense medium
            float sigma_t = nextHit.material.alpha * 2.0; // tweak 2.0

            vec3 mediumEntry = currentRay.origin + currentRay.direction * distToSurface;

            HitResult exitHit = raySceneIntersection(
                Ray(mediumEntry + epsilon * currentRay.direction, currentRay.direction),
                false
            );

            float distanceInMedium = length(exitHit.position - mediumEntry);

            // Sample free-flight distance
            float hitT = -log(max(rand(), 1e-6)) / sigma_t;

            if (hitT < distanceInMedium) {
                // Isotropic scattering event
                accum += throughput * nextHit.material.emissive; 
                vec3 scatterPoint = mediumEntry + currentRay.direction * hitT;
                throughput *= exp(-sigma_t * hitT) * nextHit.material.albedo;

                vec3 newDir = normalize(vec3(
                    2.0 * rand() - 1.0,
                    2.0 * rand() - 1.0,
                    2.0 * rand() - 1.0
                ));

                currentRay = Ray(scatterPoint + epsilon * newDir, newDir);
                continue;
            }

            // No scattering: ray exits medium
            throughput *= exp(-sigma_t * distanceInMedium);

            currentRay = Ray(exitHit.position + epsilon * currentRay.direction, currentRay.direction);
            continue;
        }

        if (!nextHit.hit) {
            accum += throughput * iBackground;
            break;
        }

        // Surface hit: accumulate emission
        accum += throughput * nextHit.material.emissive;

        // Surface scattering
        vec3 V = -currentRay.direction;
        vec3 throughputMult;
        Ray scattered = pbrBounce(V, nextHit.normal, nextHit, nextHit.material, throughputMult);
        throughput *= throughputMult;

        // Russian roulette
        float survivalProb = min(1.0, dot(throughput, vec3(0.299, 0.587, 0.114)));
        if (rand() > survivalProb)
            break;
        throughput /= max(1e-6, survivalProb);

        // Advance ray to scattered ray and loop
        currentRay = scattered;
    }
    col = accum;
}

void main() {
    rng.state = uvec2(
        uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + floatBitsToUint(iTime) * 26699u,
        uint(gl_FragCoord.x) * 2713u + uint(gl_FragCoord.y) * 1619u + floatBitsToUint(iTime) * 11003u
    );
    rng.inc = uvec2(0x9E3779B1u, 0x00000000u);
    int SAMPLES = max(iSamples, 1);

    vec2 uv = (gl_FragCoord.xy / iResolution) * 2.0 - 1.0;
    uv.x *= iResolution.x / max(iResolution.y, 1.0); 

    // Simple pinhole camera
    vec3 camPos  = vec3(iTime, iTime - 5, -5.0);   // back and slightly above
    vec3 camTarget = vec3(0.0, 2.5, 0.0);  // look at center-ish
    vec3 camDir = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(camDir, vec3(0.0, 1.0, 0.0)));
    vec3 camUp = cross(camRight, camDir);

    float fov = radians(45.0);
    float z = 1.0 / tan(fov * 0.5);

    // Length where the ray hits the focal plane, controls depth of field strength.
    // This does NOT control the field of view - FOV is handled by 'z'.
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

        HitResult result = raySceneIntersection(ray, true);

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
