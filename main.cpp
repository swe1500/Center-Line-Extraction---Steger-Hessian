#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>

class ExtractLines{
private:
    cv::Mat Src,    //Define Mat to hold source image
            Img,    //Original image
            Dst,    //Processed image
            Dx,     //Partial derivative with respect to x
            Dy,     //Partial derivative with respect to y
            Dxx,    //Second derivate with respect to x
            Dxy,    //Second partial derivative with respect to x and y, Dxy = Dyx
            Dyy;    //Second derivative with respect to y
    std::vector<cv::Point> *DetectPoints;
    void ComputeDerivative(void);
    void ComputeHessian(void);
public:
    ExtractLines(const cv::Mat& Img);
    void imshow(void) const;
    inline ~ExtractLines(){delete DetectPoints;}
};

/**
 * @brief  Compute first and second partial derivative matrixes.
 *         Call ComputeHessian() after all the derivative matrixes
 *         has been computed.
**/
void ExtractLines::ComputeDerivative(void){

    cv::Mat Mask_Dx = (cv::Mat_<float>(3, 1) <<  1,  0 , -1);
    cv::Mat Mask_Dy = (cv::Mat_<float>(1, 3) <<  1,  0 , -1);
    cv::Mat Mask_Dxx = (cv::Mat_<float>(3, 1) << 1, -2 ,  1);
    cv::Mat Mask_Dyy = (cv::Mat_<float>(1, 3) << 1, -2 ,  1);
    cv::Mat Mask_Dxy = (cv::Mat_<float>(2, 2) << 1, -1 , -1, 1);

    cv::filter2D(Dst, Dx, CV_32FC1, Mask_Dx);
    cv::filter2D(Dst, Dy, CV_32FC1, Mask_Dy);
    cv::filter2D(Dst, Dxx, CV_32FC1, Mask_Dxx);
    cv::filter2D(Dst, Dxy, CV_32FC1, Mask_Dxy);
    cv::filter2D(Dst, Dyy, CV_32FC1, Mask_Dyy);

    ComputeHessian();
}

/**
 * @brief  Compute first and second partial derivative matrixes.
 *         Call ComputeHessian() after all the derivative matrixes
 *         has been computed.
**/
void ExtractLines::ComputeHessian(void){
    for(int x = 0; x < Dst.cols; x++)
    {
        for(int y = 0; y < Dst.rows; y++)
        {
            if(Src.at<uchar>(y,x) > 10)
            {
                cv::Mat Hessian(2,2,CV_32FC1),Eigenvalue,Eigenvector;
                Hessian.at<float>(0, 0) = Dxx.at<float>(y, x);
                Hessian.at<float>(0, 1) = Dxy.at<float>(y, x);
                Hessian.at<float>(1, 0) = Dxy.at<float>(y, x);
                Hessian.at<float>(1, 1) = Dyy.at<float>(y, x);

                cv::eigen(Hessian, Eigenvalue, Eigenvector);
                double Nx, Ny;
                if(fabs(Eigenvalue.at<float>(0, 0)) >= fabs(Eigenvalue.at<float>(1, 0)))
                {
                    Nx = Eigenvector.at<float>(0, 0);
                    Ny = Eigenvector.at<float>(0, 1);
                }
                else
                {
                    Nx = Eigenvector.at<float>(1, 0);
                    Ny = Eigenvector.at<float>(1, 1);
                }
                double denominator = Dxx.at<float>(y,x) * Nx * Nx + 2 * Dxy.at<float>(y,x) * Nx * Ny + Dyy.at<float>(y,x) * Ny * Ny, t;
                if(denominator != 0.0)
                {
                    t = -(Dx.at<float>(y,x) * Nx + Dy.at<float>(y,x) * Ny) / denominator;
                    if(fabs(t * Nx) <= 0.5 && fabs(t * Ny) <= 0.5)
                    {
                        cv::Point_<float> res(x,y);
                        (*DetectPoints).push_back(res);
                    }
                }

            }
        }
    }
}

/**
 * @brief  Constructor, init the image and preprocess the input image with Gaussian filter.
           Throw error if cannot open the image.
   @param  img_name: the input file name.
**/
ExtractLines::ExtractLines(const cv::Mat& Input_img){
    Src = Input_img.clone();
    Img = Input_img.clone();
    cv::cvtColor(Src, Src, CV_BGR2GRAY);    //Convert RGB image to gray space
    Dst = Src.clone();
    cv::GaussianBlur(Dst, Dst, cv::Size(0, 0), 3, 3);
    DetectPoints = new std::vector<cv::Point>;
    ComputeDerivative();
}

/**
 * @brief  Draw detected salient points on original image using cv::circle and display the result.
**/
void ExtractLines::imshow(void) const{
    for(unsigned int i = 0; i < (*DetectPoints).size(); i++){
        cv::circle(Img, (*DetectPoints)[i], 0.3, cv::Scalar(0, 0, 200));
    }
    cv::imshow("result", Img);
    cv::waitKey(0);
}

#ifndef TEST

int main()
{
    cv::Mat Src= cv::imread("test.BMP",1);
    if(!Src.data){
        printf("fail to open the image!\n");
        return -1;
    }
    ExtractLines lines(Src);
    lines.imshow();
    return 0;

}
#endif

