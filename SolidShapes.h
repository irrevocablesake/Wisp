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
        resourceManager -> setupBuffer(geometry.VHandle, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        resourceManager -> copyDataIntoBuffer(geometry.VHandle.allocationInfo.pMappedData, geometry.vertices.data(), geometry.VHandle.size);

        //INDEX BUFFER
        geometry.IHandle.size = sizeof(uint32_t) * geometry.indices.size();
        resourceManager -> setupBuffer(geometry.IHandle, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        resourceManager -> copyDataIntoBuffer(geometry.IHandle.allocationInfo.pMappedData, geometry.indices.data(), geometry.IHandle.size);

        return geometry;
	}
};