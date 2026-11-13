#pragma once

#include "resourceManager.h"

#include<vector>
#include<glm/glm.hpp>

class SolidShapes {
private:
    const float PI = 3.14159265359f;
    ResourceManager* resourceManager;
public: 
    void initialize(ResourceManager&_resourceManager) {
        resourceManager = &_resourceManager;
    }

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
    };

	struct Geometry {
		std::vector< Vertex > vertices;
		std::vector< uint32_t > indices;

        ResourceManager::BufferHandle VHandle;
        ResourceManager::BufferHandle IHandle;
	};

	Geometry generateSphere( uint32_t segments, uint32_t rings, float radius ) {
		Geometry geometry;

        for (uint32_t y = 0; y <= rings; ++y)
        {
            float v = float(y) / rings;
            float phi = v * PI;

            for (uint32_t x = 0; x <= segments; ++x)
            {
                float u = float(x) / segments;
                float theta = u * 2.0f * PI;

                glm::vec3 position{
                    std::sin(phi) * std::cos(theta),
                    std::cos(phi),
                    std::sin(phi) * std::sin(theta)
                };

                position = position * radius;

                geometry.vertices.push_back({
                    position,
                    glm::normalize(position)
                });
            }
        }

        for (uint32_t y = 0; y < rings; ++y)
        {
            for (uint32_t x = 0; x < segments; ++x)
            {
                uint32_t a = y * (segments + 1) + x;
                uint32_t b = a + segments + 1;

                geometry.indices.push_back(a);
                geometry.indices.push_back(a + 1);
                geometry.indices.push_back(b);

                geometry.indices.push_back(a + 1);
                geometry.indices.push_back(b + 1);
                geometry.indices.push_back(b);
            }
        }

        //VERTEX BUFFER
        geometry.VHandle.size = sizeof(Vertex) * geometry.vertices.size();
        resourceManager->setupBuffer(geometry.VHandle, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, true, false );
        resourceManager ->uploadData(geometry.VHandle, geometry.vertices.data(), geometry.VHandle.size);

        //INDEX BUFFER
        geometry.IHandle.size = sizeof(uint32_t) * geometry.indices.size();
        resourceManager -> setupBuffer(geometry.IHandle, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, false, true, false );
        resourceManager -> uploadData(geometry.IHandle, geometry.indices.data(), geometry.IHandle.size);

        return geometry;
	}
    Geometry generateWireframe(glm::vec3 min, glm::vec3 max) {
        Geometry geometry;

        const glm::vec3& mn = min;
        const glm::vec3& mx = max;

        // 8 corners of the bounding box
        geometry.vertices = {
            { {mn.x, mn.y, mn.z}, glm::vec3(0.0f) }, // 0
            { {mx.x, mn.y, mn.z}, glm::vec3(0.0f) }, // 1
            { {mx.x, mx.y, mn.z}, glm::vec3(0.0f) }, // 2
            { {mn.x, mx.y, mn.z}, glm::vec3(0.0f) }, // 3
            { {mn.x, mn.y, mx.z}, glm::vec3(0.0f) }, // 4
            { {mx.x, mn.y, mx.z}, glm::vec3(0.0f) }, // 5
            { {mx.x, mx.y, mx.z}, glm::vec3(0.0f) }, // 6
            { {mn.x, mx.y, mx.z}, glm::vec3(0.0f) }  // 7
        };

        // 12 edges, each represented by 2 indices
        geometry.indices = {
            // Bottom face
            0, 1,
            1, 2,
            2, 3,
            3, 0,

            // Top face
            4, 5,
            5, 6,
            6, 7,
            7, 4,

            // Vertical edges
            0, 4,
            1, 5,
            2, 6,
            3, 7
        };

        geometry.VHandle.size = geometry.vertices.size() * sizeof(Vertex);
        resourceManager->setupBuffer(
            geometry.VHandle,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            false, false, true
        );
        resourceManager->uploadData(
            geometry.VHandle,
            geometry.vertices.data(),
            geometry.VHandle.size
        );

        geometry.IHandle.size = geometry.indices.size() * sizeof(uint32_t);
        resourceManager->setupBuffer(
            geometry.IHandle,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            false, false, true
        );
        resourceManager->uploadData(
            geometry.IHandle,
            geometry.indices.data(),
            geometry.IHandle.size
        );

        return geometry;
    }
};