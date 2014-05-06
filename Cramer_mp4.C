// MP3
// PHYS 453: Pattern Recognition
//
// Travis Cramer

//Most variables and functions named in the file Dr. Daugherity gave us.
class NeuralNetwork {
public:
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

	NeuralNetwork();
	~NeuralNetwork();
	void FeedForward();
	void TrainSample(double * t);
	void UpdateWeights();
	double eta();
	double SetEta(double e);
	double GetSampleError(double * t, double * inputs);
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
	b = 2.0/3.0;
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
		for (int i = 0; i < in; i++) {
			dwji[j][i] = 0;
		}
	}
	
	dwkj = new double*[out];
	for (int k = 0; k < out; k++) {
		dwkj[k] = new double[hid];
		for (int j = 0; j < hid; j++) {
			dwkj[k][j] = 0;
		}
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

	//The d
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
	
	for (int j = 0; j < hid-1; j++) {
		delete []wji;
	}
	delete []wji;

	for (int k = 0; k < out; k++) {
		delete []wkj;
	}
	delete []wkj;

	for (int j = 0; j < hid-1; j++) {
		delete []dwji;
	}
	delete []dwji;

	for (int k = 0; k < out; k++) {
		delete []dwkj;
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

		//cout << "Net J for " << j << " is " << result << endl;
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
			//cout << "\tResult so far: " << result << " at wkj at k=" << k << " and j=" << j << " is "<< wkj[k][j] << " yj: " << y[j] << endl;
		}
		//cout << "net_k for k=" << k << ": " << result << endl;
		netk[k] = result;
	}

	//Set the values of the Output layer
	for (int k = 0; k < out; k++) {
		z[k] = f(netk[k]);
	}
}

//Train a sample for given inputs.
void NeuralNetwork::TrainSample(double * t) {
	

	//FeedForward just because...
	FeedForward();
	//Compute Delta K, sensitivity
	for (int k = 0; k < out; k++) {
		delk[k] = df(netk[k]) * (t[k]-z[k]);
	}
	//Compute Delta W_KJ, the updated weights
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			dwkj[k][j] += eta * delk[k] * y[j];
		}
	}

	//Compute Delta J, sensitivity
	for (int j = 0; j < hid-1; j++) {
		delj[j] = 0.0;
		for (int k = 0; k < out; k++) {
			delj[j] += wkj[k][j] * delk[k];
		}
		delj[j] *= df(netj[j]);
	}

	//computer Delta W_JI, updated weights
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			dwji[j][i] += eta * x[i] * delj[j];
		}
	}

	//increment Training counter
	trainCount++;
}

//Applies current training, delta W becomes W.
void NeuralNetwork::UpdateWeights() {
	//copy dwji to wji,
	//cout << "updating wji\n";
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			wji[j][i] += dwji[j][i];
		}
	}
	//cout << "updating wkj\n";
	// and dwkj to wkj
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			wkj[k][j] += dwkj[k][j];
		}
	}

	//cout << "updating dwji\n";
	for (int j = 0; j < hid-1; j++) {
		for (int i = 0; i < in; i++) {
			dwji[j][i] = 0;
		}
	}

	//cout << "updating dwkj\n";
	for (int k = 0; k < out; k++) {
		for (int j = 0; j < hid; j++) {
			dwkj[k][j] = 0;
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
				largest = dwji[j][i];
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

//This is the Fast Inverse Square Root Function. Aliased because it's easy to change the methodolgy that way.
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
	return a*tanh(b*net);
}

//The derivative of the activation function
double NeuralNetwork::df(double net) {
	//cout << "Net: " << net << " b: " << b << " net*b: " << net*b << endl;
	double coshv = cosh(b*net);
	return (a*b)/(pow(coshv,2));
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
void Cramer_mp4() {
	bool debug = true;
	NeuralNetwork nn(64,3,3);
	//cout << "Contructed\n";
	nn.SetEta(0.07);
	double inputs [] = {1.0, -1.0};
	nn.SetInputs(inputs);
	//cout << "\nSet Inputs\n";
	//cout << "\nFeedForward\n";
	nn.FeedForward();
	
	//nn.PrintWeights();
	//cout << "\n1st Weight Print\n";
	//nn.PrintNetwork();
	//cout << "\n1st Network Print\n";
	double train1 [] = {-0.5}; 

	//nn.SetInputs(inputs);
	//nn.TrainSample(train1);
	//nn.UpdateWeights();


	//*******************************
	// ***** FIX - My tests here ***
	//*******************************
      
	//Do a quick stocastic training
	int NUMSAMPLES = 10000;
	double threshold = 0.000001;
	std::vector<double> thresh(NUMSAMPLES); // storage for thresholds
	std::vector<double> err(NUMSAMPLES); // storage for sample error
	int iter = 0;
	for(;;) {
		double avgErr = 0;

		double i0 = gRandom->Uniform(-1,1);
		double i1 = gRandom->Uniform(-1,1);
		inputs[0] = i0;
		inputs[1] = i1;
		nn.SetInputs(inputs);
		if(inputs[0]*inputs[1]<0) train1[0] = 1.0;
		else train1[0] = -1.0;
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		inputs[0] = -1.0*i0;
		inputs[1] = i1;
		nn.SetInputs(inputs);
		if(inputs[0]*inputs[1]<0) train1[0] = 1.0;
		else train1[0] = -1.0;
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		inputs[0] = i0;
		inputs[1] = -1.0*i1;
		nn.SetInputs(inputs);
		if(inputs[0]*inputs[1]<0) train1[0] = 1.0;
		else train1[0] = -1.0;
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		inputs[0] = -1.0*i0;
		inputs[1] = -1.0*i1;
		nn.SetInputs(inputs);
		if(inputs[0]*inputs[1]<0) train1[0] = 1.0;
		else train1[0] = -1.0;
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		avgErr /= 4;

		thresh.push_back(nn.GetThreshold());
		if (thresh.size() > NUMSAMPLES && err.size() > NUMSAMPLES) {
			NUMSAMPLES = thresh.size();
		}
		if (nn.GetThreshold() < threshold) {
			nn.UpdateWeights();
			nn.FeedForward();
			err.push_back(avgErr);
			break;
		}
		nn.UpdateWeights();
		nn.FeedForward();
		err.push_back(avgErr);
		iter++;
		
		if (iter % 1000 == 0) { 
			cout << "Still alive! Threshold: " << thresh.back() << endl;
			nn.SetEta(nn.Eta() / 1.2);
		}

		//cout << "----------------------" << endl;
		//nn.PrintWeights();
		//nn.PrintNetwork();
	}

	cout << "**** Done Training ********" << endl;
	// copy and paste of plotting code below
	nn.PrintWeights();
	nn.PrintNetwork();
	cout << inputs[0] << "\t" << inputs[1] << endl;
	
	gStyle->SetPalette(1);
	c1 = new TCanvas("c1", "Machine Problem 4",800,400);
	c1->Divide(4,2);
	// drawing stuff....need to figure it out
	// histograms
	TH1D * letter1 = 
	for (int i = 0; i < nn.hid; i++) {
		c1->cd(i+1);
		h[i]->Draw("surf1");
		gPad->SetTheta(60);
		gPad->SetPhi(-45);
	}
	
	// graphs for learning curves
	double * xaxis = new double[NUMSAMPLES];
	for(int i=0; i<NUMSAMPLES; i++) xaxis[i]=i; // Make an x-axis for our graphs
	TGraph* tgthr = new TGraph(NUMSAMPLES, xaxis, &thresh[0]);
	TGraph* tgerr = new TGraph(NUMSAMPLES, xaxis, &err[0]);
	c1->cd(7);  
	gPad->SetLogy();
	tgthr->SetTitle("Threshold");
	tgthr->Draw("AL");
	c1->cd(8);  
	gPad->SetLogy();
	tgerr->SetTitle("Error");
	tgerr->SetLineColor(kBlue);
	tgerr->Draw("AL");

	c1->Update();
	c1->Print("xor.gif");

	// *** copy and paste of evaluation code below ***
	//Error for 1000 samples
	cout << "*** Testing 1000 samples..." << endl;
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
		double sol1 = (rand1*rand2 < 0)?1.0:-1.0;  // FIX - THIS HAD THE WRONG SIGNS!
		double inputs1 [] = {rand1,rand2};
		nn.SetInputs(inputs1);
		nn.FeedForward();
		if ((nn.GetResult() > 0 && sol1 > 0) || (nn.GetResult() < 0 && sol1 < 0)) {
			right++;
		} else {
			cout << "\tWRONG: input_1: " << rand1 << " input_2: " << rand2 << " Solution: " << sol1 << " actual value: " << nn.GetResult() << endl;
		}
	}
	cout << "Right: " << right << endl;

	delete []xaxis;
	return;
	//*************************************8
	// ** untouched below here **
	//*************************************8

	//Get some random training data!
	if (!debug) {
		double threshold = 0.00001;
		int iterations = 0;
		double rand1 = 1;
		double rand2 = 1;
		double sol1 = (rand1*rand2 > 0)?1.0:-1.0;

		double inputs1 [] = {rand1,rand2};
		double train2 [] = {sol1};
		while (true) {
			//cout << "setting random 1\n";
			rand1 = 1;
			//Ensure it's not 0.
			do {
				rand1 = gRandom->Uniform(-1,1);
			} while(rand1 == 0);
			
			//cout << "setting random 2\n";
			rand2 = 1;
			//Ensure it's not 0.
			do {
				rand2 = gRandom->Uniform(-1,1);
			} while(rand2 == 0);
			
			//cout << "Calculating solution\n";
			//Calculate the right answer
			sol1 = (rand1*rand2 > 0)?1.0:-1.0;

			//cout << "setting arrays\n";
			//Set up arrays for training
			inputs1[0] = rand1;
			inputs1[1] = rand2;
			train2[0] = sol1;
			
			//cout << "Setting inputs\n";			
			//Try Online training, update after each attempt.
			nn.SetInputs(inputs1);
			//cout << "Training\n" << endl;
			nn.TrainSample(train2);

			iterations++;
			//cout << "iteration " << iterations << endl;
			if (nn.GetThreshold() < threshold) {
				nn.UpdateWeights();
				break;
			}
			nn.UpdateWeights();

		}
	} else {
		for (int i = 0; i < 50; i++) {
			nn.SetInputs(inputs);
			nn.TrainSample(train1);
			nn.UpdateWeights();
		}
	}
	//cout << "\nBackPropogate on 1\n";
	
	//cout << "\nUpdated Weights\n";
	nn.SetInputs(inputs);
	nn.FeedForward();
	nn.PrintWeights();
	//cout << "\n2nd Weight Print\n";
	nn.PrintNetwork();
	//cout << "\n2nd Weight Print\n";

	if (debug) {
		cout << "Did it converge? " << (((nn.GetResult() > 0 && train1[0] > 0) || (nn.GetResult() < 0 && train1[0] < 0))?"Yes!":"No! :(") << endl;
	} else {
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
			} else {
				cout << "\tWRONG: input_1: " << rand1 << " input_2: " << rand2 << " Solution: " << sol1 << " actual value: " << nn.GetResult() << endl;
			}
		}
		cout << "Right: " << right/10 << "%" << endl;
	}
	
	cout << "\n--------------------\nDONE WITH PROGRAM!\n--------------------\n";

	//Now do pretty stuff?
	//th2d1 = new TH2D();

	c1 = new TCanvas("c1", "Machine Problem 3",800,400);
	c1->Divide(3,2);
	// drawing stuff....need to figure it out
	// histograms
	double x1, x2;
	TH2D* h[6];
	h[0] = new TH2D("h[0]", "OUTPUT", 20, -1, 1, 20, -1, 1);
	h[1] = new TH2D("h[1]", "HIDDEN 1", 20, -1, 1, 20, -1, 1);
	h[2] = new TH2D("h[2]", "HIDDEN 2", 20, -1, 1, 20, -1, 1);
	h[3] = new TH2D("h[3]", "HIDDEN 3", 20, -1, 1, 20, -1, 1);
	h[4] = new TH2D("h[4]", "HIDDEN 4", 20, -1, 1, 20, -1, 1);
	h[5] = new TH2D("h[5]", "HIDDEN 5", 20, -1, 1, 20, -1, 1);
	for(int i=1; i<=h[0]->GetNbinsX(); i++){
		for(int j=1; j<=h[0]->GetNbinsY(); j++){
			x1 = h[0]->GetXaxis()->GetBinCenter(i);
			x2 = h[0]->GetYaxis()->GetBinCenter(j);
			nn.x[1] = x1;
			nn.x[2] = x2;
			nn.FeedForward();
			h[0]->SetBinContent(i, j, nn.z[0]);
		}
	}
	for(int num=1; num<nn.hid; num++){
		for(int i=1; i<=h[num]->GetNbinsX(); i++){
	  		for(int j=1; j<=h[num]->GetNbinsY(); j++){
				x1 = h[num]->GetXaxis()->GetBinCenter(i);
				x2 = h[num]->GetYaxis()->GetBinCenter(j);
		    	nn.x[1] = x1;
				nn.x[2] = x2;
				nn.FeedForward();
				h[num]->SetBinContent(i, j, nn.y[num]);
		  	}
		}
	}

	for (int i = 0; i < nn.hid; i++) {
		c1->cd(i+1);
		h[i]->Draw("surf1");
		gPad->SetTheta(60);
		gPad->SetPhi(-45);
	}
	c1->Update();
	c1->Print("xor.gif");
}