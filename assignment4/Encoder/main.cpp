// how to use TooJpeg: creating a JPEG file
// see https://create.stephan-brumme.com/toojpeg/

#include "toojpeg.h"

// use a C++ file stream
#include <fstream>

// rename
#include <iostream>
#include <cstdio>

// use opencv to open image from file
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void myOutput(unsigned char);
double getPSNR(const Mat& , const Mat&);

// output file
std::ofstream myFile("Output.jpg", std::ios_base::out | std::ios_base::binary);


// main function
int main(int argc, char *argv[])
{
    Mat my_image = imread(argv[1], 4);
    
    // image size
    const auto width = my_image.cols;
    const auto height = my_image.rows;

    
    // RGB: one byte each for red, green, blue
    const auto bytesPerPixel = my_image.channels();// 
    cout << bytesPerPixel << endl;
    // allocate memory
    auto image = new unsigned char [width * height * bytesPerPixel];

    // create a nice color transition (replace with your code)


    for (auto y = 0; y < height; y++)
    {
        for (auto x = 0; x < width; x++)
        {
            // memory location of current pixel
            auto offset = (y * width + x) * bytesPerPixel;

            if (bytesPerPixel == 1) {
                image[offset] = my_image.at<uchar>(Point(x, y));
            }
            else {

                image[offset] = my_image.at<Vec3b>(Point(x, y))[2]; //b
                image[offset + 1] = my_image.at<Vec3b>(Point(x, y))[1]; //g
                image[offset + 2] = my_image.at<Vec3b>(Point(x, y))[0]; //r
            }
        }
    }
    
    // std::cout << typeid(my_image.at<Vec3b>(Point(0, 0))[2]).name() << '\n';
    // start JPEG compression
    // note: myOutput is the function defined in line 18, it saves the output in example.jpg
    // optional parameters:
    const bool isRGB = (bytesPerPixel == 3);  // true = RGB image, else false = grayscale
    const auto quality = 90;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
    const bool downsample = true; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
    const char* comment = ""; // arbitrary JPEG comment
    auto ok = TooJpeg::writeJpeg(myOutput, image, width, height, isRGB, quality, downsample, comment);

    delete[] image;

    myFile.close();


    // rename jpg file
    std::string s = argv[1];
    std::string delimiter = ".";

    size_t pos = 0;
    std::string token;
    if ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        s.erase(0, pos + delimiter.length());
    }

    const char *oldname = "Output.jpg";

    std::string end_name = ".jpg";
    std::string full_name = token + end_name;
    const char *newname = full_name.c_str();


    if (rename(oldname, newname) != 0)
        perror("Error renaming file");
    else
        cout << "File renamed successfully" << endl;

    double psnr;

    Mat my_jpeg_image = imread(newname, 4);
    psnr = getPSNR(my_image, my_jpeg_image);
    cout << "PSNR: " << psnr << endl;


    // error => exit code 1
    return ok ? 0 : 1;
}


// write a single byte compressed by tooJpeg
void myOutput(unsigned char byte)
{
    myFile << byte;
}


double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}