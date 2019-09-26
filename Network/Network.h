#pragma once
#include "Maths.h"
#include <fstream>

class Network
{
public:
	Network(int input_size,int hidden_size,int out_size);

	Matrix forward(const vector<double> &input);
	void backward(vector<double> out);
	void train(int epochs, const Matrix& input, const Matrix& output,double learning_rate);
	void save(const char* path);
	void load(const char* path);

private:
	double sigmoid(double x);
	Matrix weights1;
	Matrix weights2;
	Matrix bias1;
	Matrix bias2;
	Matrix z0;
	Matrix z1;
	Matrix z2;
	float learning_rate;
	void write(ofstream& out, const Matrix& m);
	vector<string> split(const string& s,const char);
};

