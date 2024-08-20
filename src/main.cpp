/*
 * @Author: joword 23089538@qq.com
 * @Date: 2024-08-20 21:43:40
 * @LastEditTime: 2024-08-20 23:19:38
 * @FilePath: \personal-healthcare-ai\src\main.cpp
 * @Description: opencv to recognize a jpeg
 */
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    cv::Mat img = cv::imread("../static/tulip.jpeg");
    if (img.empty())
    {
        cerr<<"无法读取图片"<<endl;
        return -1;
    }
    cv::imshow("Original Image", img);
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Gray Image", grayImage);
    cv::Mat edges;
    cv::Canny(grayImage, edges, 100, 200);
    cv::imshow("Edges", edges);
    cv::waitKey(0);

    return 0;
}
