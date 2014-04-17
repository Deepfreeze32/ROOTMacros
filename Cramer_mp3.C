// MP3
// PHYS 453: Pattern Recognition
//
// Travis Cramer

//Most variables and functions named in the file Dr. Daugherity gave us.
class NeuralNetwork {
private:
	double a;
	double b;
	//node counts
	int in;
	int hid;
	int out;

	//x,y, and z
	double * x; //input layer,
	double * y; //hidden layer
	double * z; //output layer

	//w's
	double ** wji; // input-to-hidden weights 
	double ** wkj; // hidden-to-output weights 
	
	double ** dwji; // training updates for input-to-hidden weights 
	double ** dwkj; // training updates for hidden-to-output weights 
	
	//Eta value
	double eta; //Eta is a constant

	//netvalues for training
	double * netj;
	double * netk;
	double * delj;
	double * delk;

	//counters
	int trainCount;
	int updateCount;

	bool init = false;

	//Helper functions
	double InvSqrt(double n);
	void CalcNetJ();
	void CalcNetK();
	void BackPropogateInit();
public:
	NeuralNetwork();
	~NeuralNetwork();
	void FeedForward();
	void TrainSample(double * t,double * inputs);
	void UpdateWeights();
	double eta();
	double SetEta(double e);
	double GetSampleError(double * t);
	double GetThreshold();

	double Eta();
	double SetEta(double e);

	double f(double net);
	double df(double net);

	void PrintNetwork();
	void PrintWeights();

	void SetInputs(double * inputs);
	double GetResult();
};

//Constructor. Give input and hidden node counts WITHOUT bias included. 
NeuralNetwork::NeuralNetwork(int input, int hidden, int output) {
	//set variables
	a = 1.716;
	b = 1.0/3.0;
	in = input + 1;
	hid = hidden + 1;
	out = output;

	updateCount = 0;
	trainCount = 0;
	
	//allocate memory
	x = new double[in];
	y = new double[hid];
	z = new double[out];

	netj = new double[hid-1];
	netk = new double[out];
	delj = new double[hid-1];
	delk = new double[out];

	//Bias
	x[0] = 1.0;
	y[0] = 1.0;

	wji = new double*[hid-1];
	for (int j = 0; j < hid-1; j++) {
		wji[j] = new double[in];
	}
	
	wkj = new double*[out];
	for (int k = 0; k < out; k++) {
		wkj[k] = new double[hid];
	}
	
	dwji = new double*[hid-1];
	for (int j = 0; j < hid-1; j++) {
		dwji[j] = new double[in];
	}
	
	dwkj = new double*[out];
	for (int k = 0; k < out; k++) {
		dwkj[k] = new double[hid];
	}
	
	//initialize weights.
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			do { wji[j][i] = gRandom->Uniform(-InvSqrt(in),InvSqrt(in));
			} while (wji[j][i] == 0);
		}
	}
	
	for (int k = 0; k < output; k++) {
		for (int j = 0; j < hid; j++) {
			do {wkj[k][j] = gRandom->Uniform(-InvSqrt(hid), InvSqrt(hid));
			} while (wkj[k][j] == 0);
		}
	}

	//The default
	eta = 0.1;
}

//Destructor to free memory.
NeuralNetwork::~NeuralNetwork() {
	//delete memory
	delete x;
	delete y;
	delete z;

	delete netj;
	delete netk;
	delete delk;
	delete delj;
	
	for (int j = 0; j < hid-1; j++) {
		delete []wji[];
	}
	delete []wji;

	for (int k = 0; k < out; k++) {
		delete []wkj[];
	}
	delete []wkj;

	for (int j = 0; j < hid-1; j++) {
		delete []dwji[];
	}
	delete []dwji;

	for (int k = 0; k < out; k++) {
		delete []dwkj[];
	}
	delete []dwkj;
}

//Manually specify what the inputs are.
double NeuralNetwork::SetInputs(double * inputs) {
	for (int i = 1; i < in; i++) {
		x[i] = inputs[i-1];
	}
}

//The feedforward operation. Evaluates the network with current weights.
void NeuralNetwork::FeedForward() {
	//Calculate net_J
	for (int j = 0; j < hid-1; j++) {
		double result = 0.0;
		for (int i = 0; i < in; i++) {
			result += wji[j][i]*x[i];
		}

		//`cout << "Net J for " << j << " is " << result << endl;
		netj[j] = result;
	}

	//Set the values of the hidden nodes
	for (int j = 1; j < hid; j++) {
		//cout << "f(netJ) for j " << j << " is " << f(netj[j-1]) << endl;
		y[j] = f(netj[j-1]);
	}

	//Calculate net_K
	for (int k = 0; k < out; k++) {
		double result = 0.0;
		for (int j = 0; j < hid; j++) {
			result += wkj[k][j]*y[j];
		}
		netk[k] = result;
	}

	//Set the values of the Output layer
	for (int k = 0; k < out; k++) {
		z[k] = f(netk[k]);
	}
}

//Train a sample for given inputs.
void NeuralNetwork::TrainSample(double * t, double * inputs) {
	//Set inputs just in case.
	SetInputs(inputs);

	//FeedForward just because...
	FeedForward();
	//Compute Delta K, sensitivity
	for (int k = 0; k < out; k++) {
		delk[k] = df(netk[k])*(t[k]-z[k]);
	}
	//Compute Delta W_KJ, the updated weights
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			dwkj[k][j] = eta*delk[k]*y[j];
		}
	}

	//Compute Delta J, sensitivity
	for (int j = 0; j < hid-1; j++) {
		double sum = 0.0;
		for (int k = 0; k < out; k++) {
			sum += wkj[k][j]*delk[k];
		}
		delj[j] = df(netj[j])*sum;
	}

	//computer Delta W_JI, updated weights
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			dwji[j][i] = eta*x[i]*delj[j];
		}
	}

	//increment Training counter
	trainCount++;
}

//Applies current training, delta W becomes W.
void NeuralNetwork::UpdateWeights() {
	//copy dwji to wji,
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			wji[j][i] = dwji[j][i];
		}
	}
	// and dwkj to wkj
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			wkj[k][j] = dwkj[k][j];
		}
	}

	//increment update counter
	updateCount++;
}

//Evaluates J(w) for a training set.
double NeuralNetwork::GetSampleError(double * t, double * inputs) {
	SetInputs(inputs);
	FeedForward();
	double sum = 0.0;
	for (int k = 0; k < out; k++) {
		sum += pow(t[k]-z[k], 2);
	}
	sum *= 0.5;
	return sum;
}

//Returns largest weight from dw. I think this is what the comment meant?
double NeuralNetwork::GetThreshold() {
	double largest = dwji[1][0];
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			if (dwji[j][i] > largest) {
				largest = dwji[j+1][i];
			}
		}
	}

	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			if (dwkj[k][j] > largest) {
				largest = dwkj[k][j];
			}
		}
	}

	return largest;
}

//This is the Fast Inverse Square Root Function. Aliased because it's easy to change the emthodolgy that way.
double NeuralNetwork::InvSqrt(double n) {
	return pow(sqrt(n),-1);
}

//Getter for eta.
double NeuralNetwork::Eta() {

	return eta;
}

//Setter for eta.
double NeuralNetwork::SetEta(double e) {
	eta = e;

	return eta;
}

//The activation function
double NeuralNetwork::f(double net) {
	//cout << "a: " << a << " b: " << b << " net: " << net << endl;
	return a*TMath::TanH(b*net);
}

//The derivative of the activation function
double NeuralNetwork::df(double net) {
	return (a*b)/(TMath::CosH(b*net)*TMath::CosH(b*net));
}

//Display network weights
void NeuralNetwork::PrintWeights() {
	cout << "Weights: \nFirst printing the weights from input to hidden\n";

	for (int j = 0; j < hid-1; j++) {
		cout << "Node Y_" << j+1 << " has the following weights: \n";
		for (int i = 0; i < in; i++) {
			cout << "\tFrom node X_" << i << ": " << wji[j][i] << endl;
		}
		cout << "\t\tNet_J for j=" << j << " is: " << netj[j] << endl << endl;
	}

	cout << "Now printing from hidden to output\n";

	for (int k = 0; k < out; k++) {
		cout << "Node Z_" << k << " has the following weights: \n";
		for (int j = 0; j < hid; j++) {
			cout << "\tFrom node Y_" << j << ": " << wkj[k][j] << endl;
		}
		cout << "\t\tNet_K for k=" << k << " is: " << netk[k] << endl << endl;
	}
}

//Display the network
void NeuralNetwork::PrintNetwork() {
	//Do something pretty?
	
	cout << "Network, node values for inputs (X_0 is bias): \n";
	
	//input nodes
	for(int i = 0; i < in; i++) {
		cout << "\tNode X_" << i << ": " << x[i] << endl;
	}

	cout << "Network, node values for hidden (Y_0 is bias): \n";

	//hidden nodes
	for(int i = 0; i < hid; i++) {
		cout << "\tNode Y_" << i << ": " << y[i] << endl;
	}

	cout << "Network, node values for output: \n";

	//hidden nodes
	for(int i = 0; i < out; i++) {
		cout << "\tNode Z_" << i << ": " << z[i] << endl;
	}
}

//Just a hackish way to get the output since z is private
double NeuralNetwork::GetResult() {
	return z[0];
}

//MAIN!!!!
void Cramer_mp3() {
	NeuralNetwork nn(2,3,1);
	//cout << "Contructed\n";
	
	double inputs [] = {-1.0, 1.0};
	nn.SetInputs(inputs);
	//cout << "\nSet Inputs\n";
	cout << "\nFeedForward\n";
	nn.FeedForward();
	
	nn.PrintWeights();
	//cout << "\n1st Weight Print\n";
	nn.PrintNetwork();
	//cout << "\n1st Network Print\n";
	double train1 [] = {1}; 
	nn.TrainSample(train1,inputs);
	nn.UpdateWeights();
	//Get some random training data!
	for (int i = 0; i < 50; i++) {
		double rand1 = 1;
		//Ensure it's not 0.
		do {
			rand1 = gRandom->Uniform(-1,1);
		} while(rand1 == 0);
		
		double rand2 = 1;
		//Ensure it's not 0.
		do {
			rand2 = gRandom->Uniform(-1,1);
		} while(rand2 == 0);
		
		//Calculate the right answer
		double sol1 = (rand1*rand2 > 0)?1.0:-1.0;

		//Set up arrays for training
		double inputs1 [] = {rand1,rand2};
		double train2 [] = {sol1};
		
		//Try Online training, update after each attempt.
		nn.TrainSample(train2,inputs1);
		nn.UpdateWeights();
		//nn.TrainSample(train1,inputs);
		//nn.UpdateWeights();
	}
	//cout << "\nBackPropogate on 1\n";
	
	//cout << "\nUpdated Weights\n";
	nn.SetInputs(inputs);
	nn.FeedForward();
	nn.PrintWeights();
	//cout << "\n2nd Weight Print\n";
	nn.PrintNetwork();
	//cout << "\n2nd Weight Print\n";

	cout << "Sample Error: " << nn.GetSampleError(inputs,train1) << endl;

	//Error for 1000 samples
	int right = 0;
	for (int i = 0; i < 1000; i++) {
		double rand1 = 1;
		do {
			rand1 = gRandom->Uniform(-1,1);
		} while(rand1 == 0);
		double rand2 = 1;
		do {
			rand2 = gRandom->Uniform(-1,1);
		} while(rand2 == 0);
		double sol1 = (rand1*rand2 > 0)?1.0:-1.0;
		double inputs1 [] = {rand1,rand2};
		nn.SetInputs(inputs1);
		nn.FeedForward();
		if ((nn.GetResult() > 0 && sol1 > 0) || (nn.GetResult() < 0 && sol1 < 0)) {
			right++;
		}
	}
	cout << "Right: " << right/10 << "%" << endl;
	//cout << "Did it converge? " << ((nn.GetResult() > 0)?"Yes!":"No! :(") << endl;
	cout << "\n--------------------\nDONE WITH PROGRAM!\n--------------------\n";

	//Now do pretty stuff?
	//th2d1 = new TH2D();
}