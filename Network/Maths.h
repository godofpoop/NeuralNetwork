#pragma once
#include <vector>
#include <iostream>
#include <functional>

using namespace std;

class Matrix {
public:
	Matrix() { }
	Matrix(int rows,int columns);
	Matrix(const vector<vector<double>> &in);
	int rows;
	int columns;
	Matrix operator*(const Matrix& m) const;
	Matrix operator*(const double& m) const;
	Matrix operator+(const Matrix& m) const;
	Matrix operator-(const Matrix& m) const;
	vector<double> &operator[](int k);
	vector<double> operator[](int k) const;
	Matrix dot(const Matrix& m) const;
	Matrix sigmoid();
	Matrix d_sigmoid();
	Matrix sigmoid2c();
	Matrix transpose() const;
	Matrix randomize() const;

private:
	vector<vector<double>> data;
	void sigm_c(Matrix *m,int s,int f);
	double sigm(double x);
	double d_sigm(double x);
};
class Maths
{
};
ostream& operator << (ostream& out, const Matrix& c);