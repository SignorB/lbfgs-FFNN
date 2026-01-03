#pragma once
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

class MNISTLoader {
private:
    static uint32_t reverseInt(uint32_t i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }

public:
    static Eigen::MatrixXd loadImages(const std::string& path, int max_images = 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

        uint32_t magic_number = 0;
        uint32_t number_of_images = 0;
        uint32_t n_rows = 0;
        uint32_t n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);

        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        int image_size = n_rows * n_cols;

        uint32_t count = number_of_images;
        if (max_images > 0 && max_images < (int)number_of_images) {
            count = (uint32_t)max_images;
        }

        std::cout << "Loading " << count << " images..." << std::endl;

        Eigen::MatrixXd images(image_size, count);

        for (uint32_t i = 0; i < count; ++i) {
            for (int j = 0; j < image_size; ++j) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));

                images(j, i) = (double)temp / 255.0; 
            }
        }
        return images;
    }

    static Eigen::MatrixXd loadLabels(const std::string& path, int max_images = 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

        uint32_t magic_number = 0;
        uint32_t number_of_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        uint32_t count = number_of_labels;
        if (max_images > 0 && max_images < (int)number_of_labels) {
            count = (uint32_t)max_images;
        }

        std::cout << "Loading " << count << " labels..." << std::endl;

 
        Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(10, count);

        for (uint32_t i = 0; i < count; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            if(temp < 10) {
 
                labels((int)temp, i) = 1.0;
            }
        }
        return labels;
    }
};
