// MP3
// PHYS 453: Pattern Recognition
//
// Travis Cramer

class NeuralNetwork {
private:
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

	//Helper functions
	InvSqrt(double n);
	GradW(int i, int j, double ** w);
public:
	NeuralNetwork();
	~NeuralNetwork();
	void FeedForward();
	void TrainSample(double * t);
	void UpdateWeights();
	double eta();
	double SetEta(double e);
	double GetSampleError(double * t);
	double GetThreshold();

	double f(double net);
	double df(double net);

	void PrintNetwork();
	void PrintWeights();
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

	//f(double net)

	//df(double net)

	//PrintNetwork()

	//PrintWeights()

	//Does NOT include the Bias. for example, 3 for input means three input neurones plus the bias.
	//Allocate arrays, set to 0
	
};

NeuralNetwork::NeuralNetwork(int input, int hidden, int output) {
	//set variables
	in = input;
	hid = hidden;
	out = output;
	
	//allocate memory
	x = new double[input+1];
	y = new double[hidden+1];
	z = new double[output];

	wji = new double*[hidden+1];
	for (int j = 0; j < hidden+1; j++) {
		wji[j] = new double[input+1];
	}
	
	wkj = new double*[output];
	for (int k = 0; k < output; k++) {
		wkj[k] = new double[hidden+1];
	}
	
	dwji = new double*[hidden+1];
	for (int j = 0; j < hidden+1; j++) {
		dwji[j] = new double[input+1];
	}
	
	dwkj = new double*[output];
	for (int k = 0; k < output; k++) {
		dwkj[k] = new double[hidden+1];
	}
	
	//initialize weights.
	for (int j = 0; j < hidden+1; j++) {
		for (int i = 0; i < input+1; i++) {
			wji[j][i] = gRandom->Uniform(-InvSqrt(input+1),InvSqrt(input+1));
		}
	}
	
	for (int k = 0; k < output; k++) {
		for (int j = 0; j < hidden+1; j++) {
			wkj[k][j] = gRandom->Uniform(-InvSqrt(hidden+1), InvSqrt(hidden+1));
		}
	}
}
NeuralNetwork::~NeuralNetwork() {
	//delete memory
	delete x;
	delete y;
	delete z;
	
	for (int j = 0; j < hid+1) {
		delete []wji[];
	}
	delete []wji;
	
	for (int j = 0; j < out) {
		delete []wkj[];
	}
	delete []wkj;
	
	for (int j = 0; j < hid+1) {
		delete []dwji[];
	}
	delete []dwji;
	
	for (int j = 0; j < out) {
		delete []dwkj[];
	}
	delete []dwkj;
}

//This is the Fast Inverse Square Root Function. It's magical...
double NeuralNetwork::InvSqrt(double n) {
	float number = static_cast<float>(n);

	long i;
	float x2, y;
	const float threehalfs = 1.5F;
 
	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;
	i  = 0x5f3759df - ( i >> 1 );
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );
	
	double retVal = static_cast<double>(y);
	
	return retVal;
}

double NeuralNetwork::eta() {
	return eta;
}

double NeuralNetwork::SetEta(double e) {
	eta = e;
	
	return eta;
}

double NeuralNetwork::f(double net) {
	return TMath::TanH(net);
}

double NeuralNetwork::df(double net) {
	return pow(pow(TMath::CosH(net),-1),2);
}