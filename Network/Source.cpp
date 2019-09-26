#include <iostream>
#include "Network.h"
#include <ctime>

int main()
{
	srand(time(0));
	
	// if y == 2x f(x,y) = 1
	// else f(x,y) = 0
	// 0 < x < 10
	Matrix in(100, 2);
	Matrix out(100, 1);

	for (int i = 0; i < 100; i++) {
		double x = (double)(rand() % 10);
		double y = rand() % 2 == 0 ? x*2 : (rand() % 10);

		in[i] = {x,y};
		out[i] = { (y == 2 * x ? 1.0 : 0.0) };
	}

	cout << out;
	cout << in;
	Network nn(2, 4, 1);
	
	vector<double> pred = { 2,2 };
	
	
	//nn.load("data.txt");
	nn.train(5000, in, out, 0.02);
	
	for (int i = 0; i < 10; i++) {
		double x = (double)(rand() % 10);
		double y = rand() % 2 == 0 ? x * 2 : (rand() % 10);

		vector<double> tmp = { x,y };

		cout << "Values: " << tmp[0] << "," << tmp[1] << "\nPredicted: " << nn.forward(tmp) << "\n";
	}
	//nn.save("data.txt");
	cout << nn.forward(pred) << "\n";
}