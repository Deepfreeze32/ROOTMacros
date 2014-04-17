// MP3
// PHYS 453: Pattern Recognition
//
// Travis Cramer

//Most variables and functions named in the file Dr. Daugherity gave us.
class NeuralNetwork {
private:
	const double a = 1.716;
	const double b = 1.0/3.0;
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
	int * netj;
	int * netk;
	int * delj;
	int * delk;

	//counters
	int trainCount;
	int updateCount;

	bool init = false;

	//Helper functions
	double InvSqrt(double n);
	//GradW(int i, int j, double ** w);
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
	//FeedForward

	//TrainSample(double * t)
	//calc GradW
	//trainCount++ 

	//UpdateWeights	
	//w+GradW
	//GradW=0
	//updateCount++

	//GetSampleError(double * t)
	//assume x
	//FeedForward
	//Return J

	//GetThreshold()
	//return max GradW

	//Does NOT include the Bias. for example, 3 for input means three input neurones plus the bias.
	//Allocate arrays, set to 0
	
};

//Constructor. Give input and hidden node counts WITHOUT bias included. 
NeuralNetwork::NeuralNetwork(int input, int hidden, int output) {
	//set variables
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

	//The James bond Eta value!
	eta = 0.07;
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
	
	for (int j = 0; j < hid; j++) {
		delete []wji[];
	}
	delete []wji;

	for (int k = 0; k < out; k++) {
		delete []wkj[];
	}
	delete []wkj;

	for (int j = 0; j < hid; j++) {
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
	for (int j = 0; j < hid-1; j++) {
		double result = 0.0;
		for (int i = 0; i < in; i++) {
			result += wji[j][i]*x[i];
		}
		netj[j] = result;
	}

	for (int j = 1; j < hid; j++) {
		y[j] = f(netj[j-1]);
	}

	for (int k = 0; k < out; k++) {
		double result = 0.0;
		for (int j = 0; j < hid; j++) {
			result += wkj[k][j]*y[j];
		}
		netk[k] = result;
	}

	for (int k = 0; k < out; k++) {
		z[k] = f(netk[k]);
	}
}

//Train a sample for given inputs.
void NeuralNetwork::TrainSample(double * t, double * inputs) {
	SetInputs(inputs);
	//cout << "\t\nSet inputs\n";
	FeedForward();
	//cout << "\t\nFed Forward\n";
	//Compute Delta K
	for (int k = 0; k < out; k++) {
		delk[k] = df(netk[k])*(t[k]-z[k]);
	}
	//Compute Delta W_KJ
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			dwkj[k][j] = eta*delk[k]*y[j];
		}
	}
	//
	for (int j = 0; j < hid-1; j++) {
		double sum = 0.0;
		for (int k = 0; k < out; k++) {
			sum += wkj[k][j]*delk[k];
		}
		delj[j] = df(netj[j])*sum;
	}
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			dwji[j][i] = eta*x[i]*delj[j];
		}
	}
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

	updateCount++;
}

//Evaluates J(w)
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

//Returns largest weight from dw.
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

//This is the Fast Inverse Square Root Function. It's magical...
double NeuralNetwork::InvSqrt(double n) {
	float number = static_cast<float>(n);

	long i;
	float x2, y1;
	const float threehalfs = 1.5F;
 
	x2 = number * 0.5F;
	y1  = number;
	i  = * ( long * ) &y1;
	i  = 0x5f3759df - ( i >> 1 );
	y1  = * ( float * ) &i;
	y1  = y1 * ( threehalfs - ( x2 * y1 * y1 ) );
	
	double retVal = static_cast<double>(y);
	
	return retVal;
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

//MAIN!!!!
void Cramer_mp3() {
	NeuralNetwork nn(2,2,1);
	//cout << "Contructed\n";
	nn.PrintWeights();
	//cout << "\n1st Weight Print\n";
	double inputs [] = {-0.2,0.3};
	nn.SetInputs(inputs);
	//cout << "\nSet Inputs\n";
	nn.FeedForward();
	cout << "\nFeedForward\n";
	nn.PrintNetwork();
	//cout << "\n1st Network Print\n";
	double train1 [] = {1}; 

	nn.TrainSample(train1,inputs);
	cout << "\nBackPropogate on 1\n";
	nn.UpdateWeights();
	cout << "\nUpdated Weights\n";
	nn.PrintWeights();
	//cout << "\n2nd Weight Print\n";
	nn.PrintNetwork();
	//cout << "\n2nd Weight Print\n";

	cout << "\n--------------------\nDONE WITH PROGRAM!\n--------------------\n";
}