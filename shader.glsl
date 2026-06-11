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
uniform float u_sigma_s; // scattering coefficient
uniform float u_sigma_a; // absorption coefficient
uniform float u_phase_g; // Henyey-Greenstein g parameter (-1..1)

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3 emissive;
    vec3 albedo;
    uint reflects;
    float alpha;
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

struct RNG {
    uvec2 state;
    uvec2 inc;
};

RNG rng;

uint pcg32(inout RNG rng) {
    // Simplified 32-bit LCG-based RNG to avoid using umulExtended / 64-bit ops
    // Use state.x as the 32-bit state. Update with LCG constants and a simple xorshift mix.
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

HitResult rayTriangleIntersection(Ray ray, vec3 v0, vec3 v1, vec3 v2, Material material, bool cullBackface) {
    HitResult result = HitResult(false, -1.0, vec3(0.0), vec3(0.0), material);

    vec3 rayOrigin = ray.origin;
    vec3 rayDirection = ray.direction;

    float epsilon = 0.0001;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(rayDirection, edge2);
    float a = dot(edge1, h);
    // Backface culling: a > 0 for front-face hits, a <= 0 for back-face / parallel.
    if (cullBackface && a < epsilon)
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
        vec3 normal = cross(edge1, edge2);
        if (length(normal) < epsilon)
            return result;
        result = HitResult(true, t, rayOrigin + rayDirection * t, normalize(normal), material);
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
        Material(vec3(0.0), vec3(0.0), 0u, 1.0, 1.0, 0.0)
    );

    for (int i = 0; i < numTriangles; i++) {
        Triangle tri = tris[i];

        HitResult hit = rayTriangleIntersection(
            ray,
            tri.v0, tri.v1, tri.v2,
            tri.material,
            true
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

// Sample Henyey-Greenstein phase function around direction `wo`.
vec3 samplePhaseHG(vec3 wo, float g) {
    float u1 = rand();
    float u2 = rand();
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u1;
    } else {
        float sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cosTheta = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 6.28318530718 * u2;

    vec3 up = abs(wo.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, wo));
    vec3 bitangent = cross(wo, tangent);

    return tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + wo * cosTheta;
}

// PBR bounce: probabilistically pick specular or diffuse using Fresnel weighting.
// `throughputMult` is the color to multiply into the running throughput.
Ray pbrBounce(vec3 V, vec3 N, vec3 hitPos, Material mat, out vec3 throughputMult) {
    float epsilon = 0.0001;

    // If the material lets the ray pass through, return a continued ray in
    // the incoming direction without recursing. Otherwise fall through to
    // the standard PBR scattering logic below.
    if (mat.alpha < 1.0 - epsilon) {
        if (rand() > mat.alpha) {
            // Pass through: continue along the incoming ray direction (-V)
            throughputMult = vec3(1.0);
            vec3 contDir = normalize(-V);
            vec3 origin = hitPos + contDir * 1e-4;
            return Ray(origin, contDir);
        } else {
            // Isotropic scattering: random direction in a sphere
            vec3 randomDir = normalize(vec3(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0));
            vec3 origin = hitPos;
            throughputMult = mat.albedo; // Modulate by albedo for colored transparency
            return Ray(origin, randomDir);
        }
    }

    vec3 F0 = mix(vec3(0.04), mat.albedo, mat.metallic);
    vec3 origin = hitPos + N * 1e-4;

    // Fresnel-Schlick: F(θ) = F0 + (1 - F0)(1 - cos(θ))^5
    float cosTheta = clamp(dot(V, N), 0.0, 1.0);
    float f = pow(1.0 - cosTheta, 5.0);
    vec3 fresnel = F0 + (vec3(1.0) - F0) * f;

    // Use luminance of Fresnel as specular probability (0.04 for dielectrics at normal, 1.0 for metals)
    float specProb = dot(fresnel, vec3(0.299, 0.587, 0.114));
    float denomSpec = max(specProb, epsilon);
    float denomDiff = max(1.0 - specProb, epsilon);

    if (rand() < specProb) {
        // Specular reflection
        vec3 L = reflect(-V, N);
        vec3 randomDir = randomUnitHemisphere(N);
        L = mix(L, randomDir, mat.roughness); // roughness = 0 is perfect mirror, 1 is fully random
        L = normalize(L);

        // Dielectrics reflect white, metals reflect their albedo, both modulated by Fresnel
        vec3 specColor = mix(vec3(1.0), mat.albedo, mat.metallic);
        throughputMult = fresnel * specColor / denomSpec;
        return Ray(origin, L);
    } else {
        // Diffuse reflection (only non-metals scatter diffusely)
        vec3 L = randomUnitHemisphere(N);  // cosine-weighted Lambert
        vec3 kd = (1.0 - mat.metallic) * mat.albedo;
        throughputMult = kd / denomDiff;
        return Ray(origin, L);
    }
}

void handleHit(inout vec3 col, Ray ray, HitResult hit) {
    const int MAX_BOUNCES = 6;

    Ray currentRay = ray;
    vec3 throughput = vec3(1.0);
    vec3 accum = vec3(0.0);

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        // Find next surface intersection for this ray
        HitResult nextHit = raySceneIntersection(currentRay);
        float distToSurface = nextHit.hit ? nextHit.t : 1e30;

        // Medium sampling (homogeneous medium)
        float sigma_s = max(0.0, u_sigma_s);
        float sigma_a = max(0.0, u_sigma_a);
        float sigma_t = sigma_s + sigma_a;

        bool didScatterInMedium = false;
        if (sigma_t > 1e-8) {
            float freeDist = -log(max(1e-6, 1.0 - rand())) / sigma_t;
            if (freeDist < distToSurface) {
                // Scattering event before hitting a surface
                vec3 scatterPos = currentRay.origin + currentRay.direction * freeDist;
                // albedo of the medium
                float mediumAlbedo = sigma_s / sigma_t;
                throughput *= mediumAlbedo;

                // sample new direction from phase function
                vec3 newDir = samplePhaseHG(currentRay.direction, u_phase_g);
                currentRay.origin = scatterPos + newDir * 1e-4;
                currentRay.direction = normalize(newDir);
                didScatterInMedium = true;
                // continue tracing from scattering point (counts as a bounce)
                continue;
            } else {
                // No scattering before surface: attenuate by transmittance
                throughput *= exp(-sigma_t * distToSurface);
            }
        }

        // If we reach here, either we hit a surface, or there's no medium.
        if (!nextHit.hit) {
            accum += throughput * iBackground;
            break;
        }

        // Surface hit: accumulate emission
        accum += throughput * nextHit.material.emissive;

        // Surface scattering
        vec3 V = -currentRay.direction;
        vec3 throughputMult;
        Ray scattered = pbrBounce(V, nextHit.normal, nextHit.position, nextHit.material, throughputMult);
        throughput *= throughputMult;

        // Russian roulette
        float survivalProb = min(1.0, dot(throughput, vec3(0.299, 0.587, 0.114)));
        if (rand() > survivalProb)
            break;
        throughput /= max(1e-6, survivalProb);

        // advance ray to scattered ray and loop
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
