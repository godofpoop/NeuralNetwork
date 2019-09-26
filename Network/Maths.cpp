#include "Maths.h"
#include <cassert>
#include <random>
#include <ctime>
#include <thread>

Matrix::Matrix(int rows, int columns)
{
	data = vector<vector<double>>(rows, vector<double>(columns));
	this->rows = rows;
	this->columns = columns;
}
Matrix::Matrix(const vector<vector<double>> &in)
{
	assert(in.size() != 0);
	rows = in.size();
	columns = in[0].size();
	data = in;
}

void Matrix::sigm_c(Matrix* m,int s,int f)
{
	for (int i = s; i < f; i++) {
		for (int j = 0; j < m->columns; j++) {
			m->data[i][j] = sigm(data[i][j]);
		}
	}
}
Matrix Matrix::sigmoid2c()
{
	Matrix result(rows,columns);
	thread thread0 = thread(&Matrix::sigm_c,this,&result,0,rows / 2);
	thread thread1 = thread(&Matrix::sigm_c,this,&result,rows / 2,rows);
	thread0.join();
	thread1.join();
	return result;
}
Matrix Matrix::operator*(const Matrix& m) const
{
	assert(m.rows == rows && m.columns == columns);
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = data[i][j] * m.data[i][j];
		}
	}

	return result;
}
Matrix Matrix::operator*(const double& m) const
{
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = data[i][j] * m;
		}
	}

	return result;
}
Matrix Matrix::randomize() const
{
	srand(time(0));
	Matrix result(rows, columns);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = ((double)rand() / RAND_MAX)-.5;
		}
	}

	return result;
}
Matrix Matrix::operator+(const Matrix& m) const
{
	assert(m.rows == rows && m.columns == columns);
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = data[i][j] + m.data[i][j];
		}
	}

	return result;
}
Matrix Matrix::operator-(const Matrix& m) const
{
	assert(m.rows == rows && m.columns == columns);
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = data[i][j] - m.data[i][j];
		}
	}

	return result;
}
Matrix Matrix::dot(const Matrix& m) const
{
	assert(columns == m.rows);
	double r = 0;

	Matrix result(rows, m.columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < m.columns; j++) {
			for (int k = 0; k < columns; k++) {
				r += data[i][k] * m.data[k][j];
			}
			result.data[i][j] = r;
			r = 0;
		}
	}

	return result;
}
Matrix Matrix::transpose() const
{
	Matrix result(columns, rows);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[j][i] = data[i][j];
		}
	}

	return result;
}
vector<double> &Matrix::operator[](int k)
{
	return data[k];
}
vector<double> Matrix::operator[](int k) const
{
	return data[k];
}
ostream& operator << (ostream& out, const Matrix& c)
{
	for (int i = 0; i < c.rows; i++) {
		out << "[";
		for (int j = 0; j < c.columns; j++) {
			out << c[i][j] << ",";
		}
		out << "]\n";
	}
	return out;
}
Matrix Matrix::sigmoid()
{
	Matrix result(rows, columns);
	for (int i = 0; i< rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = sigm(data[i][j]);
		}
	}

	return result;
}
Matrix Matrix::d_sigmoid()
{
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result.data[i][j] = d_sigm(data[i][j]);
		}
	}

	return result;
}
double Matrix::sigm(double x)
{
	return 1 / (1 + exp(-x));
}
double Matrix::d_sigm(double x)
{
	return exp(-x) / (pow(1 + exp(-x), 2));
}