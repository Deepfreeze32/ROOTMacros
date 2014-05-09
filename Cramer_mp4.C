// MP3
// PHYS 453: Pattern Recognition
//
// Travis Cramer

//Most variables and functions named in the file Dr. Daugherity gave us.
const double trainSize = 3.0;
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
	int GetResult();
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
	for (int i = 0; i < in; i++) {
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

		//Set output
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
			dwkj[k][j] += (eta * delk[k] * y[j]) / trainSize;
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
			dwji[j][i] += (eta * x[i] * delj[j]) / trainSize;
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
int NeuralNetwork::GetResult() {
	double max = z[0];
	int ind = 0;

	for (int k = 0; k < out; k++) {
		if (max < z[k]) {
			ind = k;
			max = z[k];
		}
	}

	return ind;
}

double * collapse(double letter[8][8],double * betterLetter) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			betterLetter[i*8+j] = letter[i][j];
		}
	}
	return betterLetter;
}

double ** noisify(double letter[8][8], double newLetter[8][8]) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			newLetter[i][j] = letter[i][j] + gRandom->Uniform(-0.5,0.5);;
		}
	}
	return newLetter;
}

double ** noisifyTest(double letter[8][8]) {
	double ** newLetter = new double*[8];
	for (int i = 0; i < 8; i++) {
		newLetter[i] = new double*[8];
		for (int j = 0; j < 8; j++) {
			newLetter[i][j] = letter[i][j] + gRandom->Uniform(-0.5,0.5);;
		}
	}
	return newLetter;
}

//MAIN!!!!
void Cramer_mp4() {
	bool debug = true;

	const int in = 64;
	const int hid = 3;
	int out = 3;
	NeuralNetwork nn(in,hid,out);
	double newLetter[8][8];
	double betterLetter[64];

	double letterA[8][8] = {{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0}};
	double letterC[8][8] = {{1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},{1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0}};
	double letterU[8][8] = {{1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{-1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0},{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}};
	
	//nn.SetEta(0.07);
	
	double * inputs = collapse(letterA,betterLetter);
	cout << "Inputing A\n";
	nn.SetInputs(inputs);
	cout << "Feeding A\n";
	nn.FeedForward();
	cout << "Setting results\n";
	double train1 [3] = {1.0,0.0,0.0}; 
	cout << "Training\n";
	nn.TrainSample(train1);

	nn.SetInputs(collapse(letterC,betterLetter));
	cout << "Feeding C\n";
	nn.FeedForward();
	cout << "Setting results\n";
	train1[0] = 0.0;
	train1[1] = 1.0;
	train1[2] = 0.0; 
	cout << "Training\n";
	nn.TrainSample(train1);

	nn.SetInputs(collapse(letterU,betterLetter));
	cout << "Feeding U\n";
	nn.FeedForward();
	cout << "Setting results\n";
	train1[0] = 0.0;
	train1[1] = 0.0;
	train1[2] = 1.0; 
	cout << "Training\n";
	nn.TrainSample(train1);
	nn.UpdateWeights();


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

		inputs = collapse(noisify(letterA,newLetter),betterLetter);
		nn.SetInputs(inputs);
		train1[0] = 1.0;
		train1[1] = 0.0;
		train1[2] = 0.0; 
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		inputs = collapse(noisify(letterC,newLetter),betterLetter);
		nn.SetInputs(inputs);
		train1[0] = 0.0;
		train1[1] = 1.0;
		train1[2] = 0.0; 
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		inputs = collapse(noisify(letterU,newLetter),betterLetter);
		nn.SetInputs(inputs);
		train1[0] = 0.0;
		train1[1] = 0.0;
		train1[2] = 1.0; 
		nn.TrainSample(train1);
		avgErr += nn.GetSampleError(train1,inputs);

		avgErr /= 3;

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
	//cout << inputs[0] << "\t" << inputs[1] << endl;
	
	gStyle->SetPalette(1);
	c1 = new TCanvas("c1", "Machine Problem 4",800,400);
	c1->Divide(3,2);
	// drawing stuff....need to figure it out
	// histograms
	TH2D * h[4];
	h[0] = new TH2D("h[0]","A",8, -1.5,1.5,8,-1.5,1.5);
	h[1] = new TH2D("h[1]","C",8, -1.5,1.5,8,-1.5,1.5);
	h[2] = new TH2D("h[2]","U",8, -1.5,1.5,8,-1.5,1.5);
	h[3] = new TH2D("h[3]","Sample Noise on A", 8, -1.5, 1.5, 8, -1.5, 1.5);
	double ** sampleNoise = noisifyTest(letterA);
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			h[0]->GetXaxis()->GetBinCenter(i+1);
			h[0]->GetYaxis()->GetBinCenter(j+1);
			h[0]->SetBinContent(i+1, j+1, letterA[i][j]);

			h[1]->GetXaxis()->GetBinCenter(i+1);
			h[1]->GetYaxis()->GetBinCenter(j+1);
			h[1]->SetBinContent(i+1, j+1, letterC[i][j]);

			h[2]->GetXaxis()->GetBinCenter(i+1);
			h[2]->GetYaxis()->GetBinCenter(j+1);
			h[2]->SetBinContent(i+1, j+1, letterU[i][j]);

			h[3]->GetXaxis()->GetBinCenter(i+1);
			h[3]->GetYaxis()->GetBinCenter(j+1);
			h[3]->SetBinContent(i+1, j+1, sampleNoise[i][j]);
		}
	}

	for (int i = 0; i < 3; i++) {
		c1->cd(i+1);
		h[i]->Draw("surf1");
		gPad->SetTheta(270);
		gPad->SetPhi(-90);
	}

	// graphs for learning curves
	double * xaxis = new double[NUMSAMPLES];
	for(int i=0; i<NUMSAMPLES; i++) xaxis[i]=i; // Make an x-axis for our graphs
	TGraph* tgthr = new TGraph(NUMSAMPLES, xaxis, &thresh[0]);
	TGraph* tgerr = new TGraph(NUMSAMPLES, xaxis, &err[0]);
	c1->cd(4);  
	gPad->SetLogy();
	tgthr->SetTitle("Threshold");
	tgthr->Draw("AL");
	c1->cd(5);  
	gPad->SetLogy();
	tgerr->SetTitle("Error");
	tgerr->SetLineColor(kBlue);
	tgerr->Draw("AL");

	c1->cd(6);
	h[3]->Draw("surf1");
	gPad->SetTheta(270);
	gPad->SetPhi(-90);

	c1->Update();
	c1->Print("OCR.gif");

	cout << "*** Testing 1000 letter samples... ***" << endl;
	int right = 0;
	for (int i = 0; i < 1000; i++) {
		inputs = collapse(noisify(letterA,newLetter),betterLetter);
		nn.SetInputs(inputs);
		nn.FeedForward();
		if (nn.GetResult() == 0) {
			right++;
		} else {
			cout << "WRONG: was given A, got: " << ((nn.GetResult() == 1)?"C":"U") << " instead";
		}

		inputs = collapse(noisify(letterC,newLetter),betterLetter);
		nn.SetInputs(inputs);
		nn.FeedForward();
		if (nn.GetResult() == 1) {
			right++;
		} else {
			cout << "WRONG: was given C, got: " << ((nn.GetResult() == 0)?"A":"U") << " instead";
		}

		inputs = collapse(noisify(letterU,newLetter),betterLetter);
		nn.SetInputs(inputs);
		nn.FeedForward();
		if (nn.GetResult() == 2) {
			right++;
		} else {
			cout << "WRONG: was given U, got: " << ((nn.GetResult() == 0)?"A":"C") << " instead";
		}
	}

	cout << "Percentage right: " << ((right/3000)*100) << "%" << endl;
	
	TH2D * weights[hid];
	for (int j = 0; j < hid; j++) {
		char buff[100];
		TString name("weights[");
		name += j;
		name += "]";
		TString desc("weights for node Y_");
		desc += (j+1);
		weights[j] = new TH2D(name, desc, 8, -1.5, 1.5, 8, -1.5, 1.5);
		for (int i = 1; i < in+1; i++) {
			weights[j]->GetXaxis()->GetBinCenter(i);
			weights[j]->GetYaxis()->GetBinCenter(j+1);
			weights[j]->SetBinContent(i, j+1, nn.wji[j][i]);
		}
	}

	TCanvas * wg[hid];
	for (int i = 0; i < hid; i++) {
		TString name("wg[");
		name +=i;
		name +="]";
		TString desc("weight graph ");
		desc += (i+1);
		wg[i] = new TCanvas(name,desc,400,400);
		weights[i]->Draw("surf1");
		gPad->SetTheta(270);
		gPad->SetPhi(-90);
	}
	
	
	delete []xaxis;
	for (int i = 0; i < 8; i++) {
		delete []sampleNoise[i];
	}
	delete []sampleNoise;
	return;
}