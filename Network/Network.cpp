#include "Network.h"
#include <chrono>
#include <sstream>
#include <thread>

Network::Network(int input_size, int hidden_size,int out_size)
{
	weights1 = Matrix(input_size, hidden_size);
	weights2 = Matrix(hidden_size, out_size);
	bias1 = Matrix(1, hidden_size);
	bias2  = Matrix(1, out_size);

	weights1 = weights1.randomize();
	weights2 = weights2.randomize();
	bias1 = bias1.randomize();
	bias2 = bias2.randomize();

	learning_rate = .02;
}
Matrix Network::forward(const vector<double> &in)
{
	z0 = Matrix({in});
	z1 = (z0.dot(weights1) + bias1).sigmoid();
	z2 = (z1.dot(weights2) + bias2).sigmoid();

	return z2;
}
void Network::backward(vector<double> output)
{ 
	Matrix out = Matrix({ output });
	Matrix db2 = (z2 - out) * ((z1.dot(weights2) + bias2).d_sigmoid());
	Matrix db1 = db2.dot(weights2.transpose()) * (z0.dot(weights1) + bias1).d_sigmoid();

	Matrix dw2 = z1.transpose().dot(db2);
	Matrix dw1 = z0.transpose().dot(db1);

	weights1 = weights1 - (dw1 * learning_rate);
	weights2 = weights2 - (dw2 * learning_rate);
	bias1 = bias1 - (db1 * learning_rate);
	bias2 = bias2 - (db2 * learning_rate);
}
void Network::write(ofstream& out, const Matrix& m)
{
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.columns; j++) {
			out << m[i][j];
			if (!(i == m.rows - 1 && j == m.columns - 1)) out << ",";
		}
	}
	out << "\n";
}
void Network::train(int epochs, const Matrix& input, const Matrix& output, double learning_rate)
{
	cout << "\n" << input[0].size() << "------>" << weights1[0].size() << "------->" << output[0].size() << "\n\n";
	double time = (double)clock();
	this->learning_rate = learning_rate;
	for(int i = 0;i<epochs;i++){
		for (int j = 0; j < input.rows; j++) {
			forward(input[j]);
			backward(output[j]);
			if(i%(epochs/100) == 0) cout << "\r" << ((double)i / epochs) * 100 << " %";
		}
	}
	cout << "\nFinished training in " << (clock() - time)*.001 << "s \n";
}
void Network::save(const char* path)
{
	ofstream out(path,ios::out);
	
	write(out, weights1);
	write(out, weights2);
	write(out, bias1);
	write(out, bias2);
	
	out.close();
}
void Network::load(const char* path)
{
	//janky file loader
	ifstream in(path);
	string s;

	in >> s;
	vector<string> d = split(s, ',');
	int p = 0;
	for (int i = 0; i < weights1.rows; i++) {
		for (int j = 0; j < weights1.columns; j++) {
			weights1[i][j] = stod(d[p++]);
		}
	}
	in >> s;
	d = split(s, ',');
	p = 0;
	for (int i = 0; i < weights2.rows; i++) {
		for (int j = 0; j < weights2.columns; j++) {
			weights2[i][j] = stod(d[p++]);
		}
	}
	in >> s;
	d = split(s, ',');
	p = 0;
	for (int i = 0; i < bias1.rows; i++) {
		for (int j = 0; j < bias1.columns; j++) {
			bias1[i][j] = stod(d[p++]);
		}
	}
	in >> s;
	d = split(s, ',');
	p = 0;
	for (int i = 0; i < bias2.rows; i++) {
		for (int j = 0; j < bias2.columns; j++) {
			bias2[i][j] = stod(d[p++]);
		}
	}
	in.close();
}
vector<string> Network::split(const string& s,const char c)
{
	vector<string> result;
	result.push_back("");
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == c) {
			result.push_back("");
			continue;
		}
		result[result.size() - 1] += s[i];
		
	}
	return result;
}
double Network::sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}