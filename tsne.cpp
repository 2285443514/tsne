#include<iostream>
#include<math.h>
#include<fstream>
#include<sstream>
#include<vector>
#include"json/json.h"
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

pair<double, MatrixXd> Hbeta(MatrixXd D, double beta = 1.0) {
    pair<double, MatrixXd> result;
    double H;
    MatrixXd P;
    
    P = ((D).array() * -beta).matrix();
    P = P.array().exp().matrix();
    double sumP = P.sum();
    H = log(sumP) + beta * (D.array() * P.array()).sum() / sumP;
    P = (P.array() / sumP).matrix();
    result = make_pair(H, P);
    return result;
}

MatrixXd x2p(MatrixXd X, double tol = 1e-5, double perplexity = 30.0) {

    int n = X.rows();
    int d = X.cols();

    Matrix<double, Dynamic, Dynamic> sum_X = X.array().square().matrix();
    sum_X = sum_X.rowwise().sum().eval();

    //cout <<"sum_X"<< sum_X << endl;

    Matrix<double, Dynamic, Dynamic> xxt = X * X.transpose();

    Matrix<double, Dynamic, Dynamic> temp = xxt * -2;

    VectorXd sum_X_vec = VectorXd::Zero(n);
    for (int i = 0; i < n; i++) { sum_X_vec(i) = sum_X(i, 0); }

    temp = temp.colwise() + sum_X_vec;

    Matrix<double, Dynamic, Dynamic> tempT = temp.transpose().eval();

    Matrix<double, Dynamic, Dynamic> D = tempT.colwise() + sum_X_vec;
   
    //cout << "D::" << endl << D << endl << endl;

    Matrix<double, Dynamic, Dynamic> P = MatrixXd::Zero(n, n);

    Matrix<double, Dynamic, Dynamic> beta = MatrixXd::Ones(1, n);
    double logU = log(perplexity);

    //迭代所有点
    for (int i = 0; i < n; i++) {

        double betamin = DBL_MIN;
        double betamax = DBL_MAX;

        Matrix<double, Dynamic, Dynamic> Di = MatrixXd::Zero(1, n - 1);
        for (int j = 0; j < n-1; j++) {
            if (j < i) {
                Di(0, j) = D(i, j);
            }
            else {
                Di(0, j) = D(i, j + 1);
            }
        }
        pair<double, MatrixXd> result = Hbeta(Di, beta(0, i));
        double H = result.first;
        MatrixXd thisP = result.second;

        //选择高斯参数
        double Hdiff = H - logU;
        int tries = 0;
        while (abs(Hdiff) > tol && tries < 50) {
            if (Hdiff > 0) {
                betamin = beta(0, i);
                if (betamax == DBL_MAX || betamax == DBL_MIN) {
                    beta(0, i) = beta(0, i) * 2;
                }
                else {
                    beta(0, i) = (beta(0, i) + betamax) / 2;
                }
            }
            else {
                betamax = beta(0, i);
                if (betamin == DBL_MAX || betamin == DBL_MIN) {
                    beta(0, i) = beta(0, i) / 2;
                }
                else {
                    beta(0, i) = (beta(0, i) + betamin) / 2;
                }
            }
            pair<double, MatrixXd> result = Hbeta(Di, beta(0, i));
            H = result.first;
            thisP = result.second;
            Hdiff = H - logU;
            tries++;
        }

        //更新当前点的P矩阵
        for (int j = 0; j < n-1; j++) {
            if (j<i) {
                P(i, j) = thisP(0, j);
            }
            else {
                P(i, j+1) = thisP(0, j);
            }
        }
    }

    //cout << "P_pos" << &P(0, 0) << " " << &P(1, 0)<<endl<<endl;

    cout << "beta" << beta << endl << endl;
    return P;
}

double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if (phase == 0) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

MatrixXd tsne(MatrixXd X, double perplexity = 30.0, int maxIter = 1000, int no_dims = 2) {
    int n = X.rows();
    int d = X.cols();
    int max_iter = maxIter;
    double initial_momentum = 0.5;
    double final_momentum = 0.8;
    double eta = 500;
    double min_gain = 0.01;
    Matrix<double, Dynamic, Dynamic> Y = MatrixXd::Zero(n, no_dims);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < no_dims; j++) {
            Y(i, j) = gaussrand();
        }
    }
    Matrix<double, Dynamic, Dynamic> dY = MatrixXd::Zero(n, no_dims);
    Matrix<double, Dynamic, Dynamic> iY = MatrixXd::Zero(n, no_dims);
    Matrix<double, Dynamic, Dynamic> gains = MatrixXd::Ones(n, no_dims);

    Matrix<double, Dynamic, Dynamic> P = x2p(X, 1e-5, perplexity);

    //cout <<"P::"<<endl << P << endl;

    P = P + P.transpose().eval();
    P = (P.array() / P.sum()).matrix();
    P = P * 4;
    P = P.cwiseMax(1e-5);

    //cout << Y << endl << endl;

    //迭代
    for (int iter = 0; iter < max_iter; iter++) {

        Matrix<double, Dynamic, Dynamic> sum_Y = (Y.array()* Y.array()).matrix();

        //cout << sum_Y << endl << endl;

        sum_Y = sum_Y.rowwise().sum().eval();

        //cout << sum_Y << endl << endl;


        Matrix<double, Dynamic, Dynamic> num = -2. * Y * Y.transpose();

        VectorXd sum_Y_vec = VectorXd::Zero(n);
        for (int i = 0; i < sum_Y.rows(); i++) { sum_Y_vec(i)= sum_Y(i, 0); }

        Matrix<double, Dynamic, Dynamic> tempSum = num.colwise() + sum_Y_vec;
        tempSum.transposeInPlace();
        tempSum = tempSum.colwise() + sum_Y_vec;



        num = (MatrixXd::Ones(n, n).array() / (tempSum.array() + 1)).matrix();
        num.diagonal() = MatrixXd::Zero(n, n).diagonal();

        Matrix<double, Dynamic, Dynamic> Q = num / num.sum();
        Q = Q.cwiseMax(1e-12);
        //cout << Q;

        Matrix<double, Dynamic, Dynamic> PQ = (P.array() - Q.array()).matrix();

        //计算梯度
        for (int i = 0; i < n; i++) {
            Matrix<double, Dynamic, Dynamic> temp = (PQ.row(i).array() * num.row(i).array()).matrix();

            VectorXd cur_Y_vec = VectorXd::Zero(no_dims);
            for (int j = 0; j < no_dims; j++) { cur_Y_vec(j) = Y(i, j); }

            Matrix<double, Dynamic, Dynamic> temp2 = (Y * -1).rowwise() + cur_Y_vec.transpose();

            Matrix<double, Dynamic, Dynamic> temp3 = MatrixXd::Zero(no_dims, n);

            for (int j = 0; j < no_dims; j++) { temp3.row(j) = temp.row(0); }

            Matrix<double, Dynamic, Dynamic> temp4 = (temp3.transpose().array() * temp2.array()).colwise().sum().matrix();

            dY.row(i) = temp4.row(0);
        }

        //更新Y
        double momentum;
        if (iter < 20) {
            momentum = initial_momentum;
        }
        else {
            momentum = final_momentum;
        }
        
        //计算增益
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < no_dims; j++) {
                if ((dY(i, j) > 0) == (iY(i, j) > 0)) {
                    gains(i, j) = gains(i, j) + 0.2;
                }
                else {
                    gains(i, j) = gains(i, j) * 0.2;
                }
                if (gains(i, j) < min_gain) {
                    gains(i, j) = min_gain;
                }
            }
        }

        iY = iY.array() * momentum - eta * (gains.array() * dY.array());

        Y = Y + iY;

        //去中心化
        Y = Y.rowwise() - Y.colwise().mean();
        if (iter == 100){
            P = P * 0.25;
        }
    }
    
    return Y;
}

MatrixXd readCSV() {
    ifstream inFile("D:\\iris.csv");
    string lineStr;
    vector<vector<double>> strArray;
    while (getline(inFile, lineStr)) {
        stringstream ss(lineStr);
        string str;
        vector<double> lineArray;
        while (getline(ss, str, ',')) {
            double str_float;
            istringstream istr(str);
            istr >> str_float;
            lineArray.push_back(str_float);
        }
        strArray.push_back(lineArray);
    }
    int n = strArray.size();
    int m = strArray[0].size();
    Matrix<double, Dynamic, Dynamic> X = MatrixXd::Zero(n, m);
    for (int i = 0; i < n;i++) {
        for (int j = 0; j < m; j++) {
            X(i, j) = strArray[i][j];
        }
    }
    return X;
}

void writeToJSON(MatrixXd X,MatrixXd Y) {
    Json::Value root;

    int n = X.rows();

    int d = X.cols();

    int no_d = Y.cols();

    for (int i = 0; i < n; i++) {
        //子节点  
        Json::Value point;
        for (int j = 0; j < d; j++) {
            point["domain"].append((double)X(i, j));
        }
        for (int j = 0; j < no_d; j++) {
            point["range"].append((double)Y(i, j));
        }
        root["points"].append(point);
    }
    

    fstream f;
    f.open("D:\\test.dimreader", ios::out);
    if (!f.is_open()) {
        cout << "Open file error!" << endl;
    }
    f << root.toStyledString(); //转换为json格式并存到文件流
    f.close();
}

int main(int argc, char** argv)
{
    Matrix<double, Dynamic, Dynamic> X = readCSV();
    Matrix<double, Dynamic, Dynamic> Y = MatrixXd::Random(X.rows(),2);
    cout << "X::"<<endl << X<< endl;
    Y = tsne(X);
    writeToJSON(X, Y);
    cout <<"Y"<< Y;
    return 0;
}