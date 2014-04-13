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
	double * x;
	double * y;
	double * z;

	//w's
	double ** wji;
	double ** wkj;
	double ** dwji;
	double ** dwkj;
	
	//Eta value
	double eta;

	//netvalues
	int * netj;
	int * netk;
	int * delj;
	int * delk;

	//counters
	int trainCount;
	int updateCount;

	//Helper functions
	InvSqrt(int n);
	GradW(int i, int j, double ** w);
public:
	NeuralNetwork();
	~NeuralNetwork();
	void FeedForward();
	void TrainSample(double * t);
	void UpdateWeights();
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

}
NeuralNetwork::~NeuralNetwork() {
	delete x;
	delete y;
	delete z;
	for (int j = 0; j < hid+1) {
		delete []wji[];
	}
	delete []wji;
}

double NeuralNetwork::f(double net) {
	return TMath::TanH(net);
}

double NeuralNetwork::df(double net) {
	return pow(pow(TMath::CosH(net),-1),2);
}