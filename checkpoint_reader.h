#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Header {
    uint32_t version;
    uint32_t magic_number;
};

struct TensorInfo {
    std::string name;
    float* data;
    uint64_t size;
    std::vector<uint32_t> shape;
    std::string encoding;
};

struct FooterMetadata {
    uint64_t footer_start;
    uint64_t num_tensors;
};

class CheckpointReader {
private:
    int fd;
    void* mapped_data;
    size_t file_size;
    std::unordered_map<std::string, TensorInfo> tensor_map;

    const char* read_string(const char* ptr) {
        size_t len = strlen(ptr);
        return ptr + len + 1;
    }

public:
    CheckpointReader(const std::string& filename) {
        std::cout << "Opening file: " << filename << std::endl;
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file");
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size");
        }
        file_size = sb.st_size;
        std::cout << "File size: " << file_size << " bytes" << std::endl;

        mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file");
        }
        std::cout << "File mapped successfully" << std::endl;

        parse_file();
    }

    ~CheckpointReader() {
        if (mapped_data != MAP_FAILED) {
            munmap(mapped_data, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
        std::cout << "Checkpoint reader destroyed" << std::endl;
    }

    void parse_file() {
        const char* data = static_cast<const char*>(mapped_data);

        const Header* header = reinterpret_cast<const Header*>(data);
        std::cout << "Header:" << std::endl;
        std::cout << "  Version: " << header->version << std::endl;
        std::cout << "  Magic number: 0x" << std::hex << header->magic_number << std::dec << std::endl;
        if (header->magic_number != 0xDEADBEEF) {
            throw std::runtime_error("Invalid magic number");
        }

        data = read_string(data + sizeof(Header));

        // Align to 256 bytes
        uintptr_t current_offset = reinterpret_cast<uintptr_t>(data);
        current_offset = (current_offset + 255) & ~255;
        data = reinterpret_cast<const char*>(current_offset);
        std::cout << "Aligned to offset: 0x" << std::hex << current_offset << std::dec << std::endl;

        const FooterMetadata* footer_metadata = reinterpret_cast<const FooterMetadata*>(
            static_cast<const char*>(mapped_data) + file_size - sizeof(FooterMetadata));
        std::cout << "Footer metadata:" << std::endl;
        std::cout << "  Footer start: " << footer_metadata->footer_start << std::dec << std::endl;
        std::cout << "  Number of tensors: " << footer_metadata->num_tensors << std::endl;

        data = static_cast<const char*>(mapped_data) + footer_metadata->footer_start;
        std::cout << "Reading tensor information..." << std::endl;
        for (uint32_t i = 0; i < footer_metadata->num_tensors; ++i) {
            TensorInfo info;

            info.name = data;
            data = read_string(data);

            info.data = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(mapped_data) +
                                                  *reinterpret_cast<const uint64_t *>(data));
            data += sizeof(uint64_t);

            info.size = *reinterpret_cast<const uint64_t*>(data);
            data += sizeof(uint64_t);

            uint32_t num_dims = *reinterpret_cast<const uint32_t*>(data);
            data += sizeof(uint32_t);

            info.shape.resize(num_dims);
            memcpy(info.shape.data(), data, num_dims * sizeof(uint32_t));
            data += num_dims * sizeof(uint32_t);

            info.encoding = data;
            data = read_string(data);


            std::cout << "Tensor " << i << ":" << std::endl;
            std::cout << "  Name: " << info.name << std::endl;
            std::cout << "  Offset: 0x" << std::hex << info.data << std::dec << std::endl;
            std::cout << "  Size: " << info.size << " bytes" << std::endl;
            std::cout << "  Shape: ";
            for (uint32_t dim : info.shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  encoding: " << info.encoding << std::endl;

            tensor_map[info.name] = std::move(info);
        }

        std::cout << "Finished parsing file" << std::endl;
    }

    const std::unordered_map<std::string, TensorInfo>& get_tensor_map() const {
        return tensor_map;
    }
};
