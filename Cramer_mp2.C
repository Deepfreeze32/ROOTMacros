//The Second ROOT assignment
//Pattern Recognition
//Travis Cramer

void Cramer_mp2() {

	const int SIZE = 10;
	const int N = 3;

	// class 1 
	float x1[] = {-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03}; 
	float y1[] = {-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33}; 
	float z1[] = {-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33}; 
	
	float c1[N][SIZE] = {x1,y1,z1};

	// class 2 
	float x2[] = {-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50}; 
	float y2[] = {-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32}; 
	float z2[] = {-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31}; 

	float c2[N][SIZE] = {x2,y2,z2};

	can1 = new TCanvas("can1","Machine Problem 2",1200,400);
	can1->Divide(3,1);

	//switch to the first display
	can1->cd(1);

	/*TMatrix x(1,1);
	TMatrix u(1,1);
	TMatrix e(1,1);

	e[0][0] = x-u;
	TMatrix et(1,1);
	et.Transpose(e);
	TMatrix sigma(1,1);
	//Variance
	sigma[0][0] = 7;
	TMatrix siginv(1,1);
	siginv = sigma;
	siginv.Invert();

	TMatrix res(1,1);
	res = et*siginv*e;*/

	//Part 1
	TMatrix mu1(N,1);
	TMatrix mu2(N,1);

	//Calculuate means
	for (int i = 0; i < SIZE; i++) {
		mu1[0] += x1[i] / SIZE;
		mu1[1] += y1[i] / SIZE;
		mu1[2] += z1[i] / SIZE;

		mu2[0] += x2[i] / SIZE;
		mu2[1] += y2[i] / SIZE;
		mu2[2] += z2[i] / SIZE;
	}

	//Print for reference
	mu1.Print();
	mu2.Print();

	//Covariance matricies
	TMatrix sig1(N,N);
	TMatrix sig2(N,N);

	/*//Calculate Covaraince
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < SIZE; k++) {
				sig1[i][j] = ((c1[i][k]-mu1[i][0])*(c1[j][k]-mu1[j][0]))/SIZE;
				sig2[i][j] = ((c2[i][k]-mu2[i][0])*(c2[j][k]-mu2[j][0]))/SIZE;
			}
		}
	}*/
	
	TMatrix d(N,1);
	TMatrix t(1,N);

	for (int i = 0; i < N; i++) {
		d[0][0] = x1[i];
		d[1][0] = y1[i];
		d[2][0] = z1[i];
		sig1 = sig1+(SIZE/100.0)*((d-mu1)*t.Transpose(d-mu1));

		d[0][0] = x2[i];
		d[1][0] = y2[i];
		d[2][0] = z2[i];
		sig2 = sig2+(SIZE/100.0)*((d-mu2)*t.Transpose(d-mu2));
	}

	//Now plot the data.
	tg1 = new TGraph(SIZE,x1,y1);
	tg1->SetMarkerStyle(20);
	tg1->SetMarkerColor(kBlack);

	tg2 = new TGraph(SIZE,x2,y2);
	tg2->SetMarkerStyle(21);
	tg2->SetMarkerColor(kRed);

	tmg = new TMultiGraph("tmg","Data points!");

	tmg->Add(tg1,"P");
	tmg->Add(tg2,"P");

	tmg->Draw("A");

	//Part 2

	//Switch display
	can1->cd(2);

	//Formulas!
	tf1 = new TF1("tf1","x^2*[0]+x*[1]+[2]",-9,9);
	tf2 = new TF1("tf2","x^2*[0]+x*[1]+[2]",-9,9);

	//double par12 = -.5*(pow((mu1[0][0]),2)/sig1[0][0]+log(sig1[0][0]))+log(0.5);
	//double par22 = -.5*(pow((mu2[0][0]),2)/sig2[0][0]+log(sig2[0][0]))+log(0.5);

	//Class 1 parameters
	double par10 = -0.5/sig1[0][0];
	double par11 = mu1[0][0]/sig1[0][0];
	double par12 = -.5*(pow((mu1[0][0]),2)/sig1[0][0]+log(sig1[0][0]))+log(0.5);

	//Class 2 parameters
	double par20 = -0.5/sig2[0][0];
	double par21 = mu2[0][0]/sig2[0][0];
	double par22 = -.5*(pow((mu2[0][0]),2)/sig2[0][0]+log(sig2[0][0]))+log(0.5);

	tf1->SetParameters(par10,par11,par12);
	tf2->SetParameters(par20,par21,par22);

	tf1->SetLineColor(kBlack);
	tf2->SetLineColor(kRed);

	tf1->SetTitle("The 1D Dichotomizer");

	tf1->Draw();
	tf2->Draw("SAME");

	//Error
	int wrong = 0;

	for (int i = 0; i < SIZE; i++) {
		if (tf1->Eval(x1[i])-tf2->Eval(x1[i]) < 0) {
			wrong++;
		}
		if (tf2->Eval(x2[i])-tf1->Eval(x2[i]) < 0) {
			wrong++;
		}
	}

	double error = (100.0*wrong)/(SIZE*2);

	printf("\nError for 1D: %d%%\n",error);


	//Part 3

	//Switch display
	can1->cd(3);


}
