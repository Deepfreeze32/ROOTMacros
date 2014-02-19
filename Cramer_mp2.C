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
	
	float x2[] = {-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50}; 
	float y2[] = {-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32}; 
	float z2[] = {-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31}; 
	
	can1 = new TCanvas("can1","Machine Problem 2",1200,400);
	can1->Divide(3,1);

	//switch to the first display
	can1->cd(1);

	/////////////////////
	///////Part 1////////
	/////////////////////
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

	TMatrix d(N,1);
	TMatrix t(1,N);

	for (int i = 0; i < SIZE; i++) {
		d[0][0] = x1[i];
		d[1][0] = y1[i];
		d[2][0] = z1[i];
		sig1 = sig1+(1.0/SIZE)*((d-mu1)*t.Transpose(d-mu1));

		d[0][0] = x2[i];
		d[1][0] = y2[i];
		d[2][0] = z2[i];
		sig2 = sig2+(1.0/SIZE)*((d-mu2)*t.Transpose(d-mu2));
	}
	
	sig1.Print();
	sig2.Print();
	
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

	/////////////////////////
	//////////Part 2/////////
	/////////////////////////

	//Switch display
	can1->cd(2);

	//Formulas!
	tf1 = new TF1("tf1","x^2*[0]+x*[1]+[2]",-9,9);
	tf2 = new TF1("tf2","x^2*[0]+x*[1]+[2]",-9,9);

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

	//Error Calculation
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

    tf3 = new TF2("tf3","x^2*[0]+x*y*[1]+x*y*[2]+y^2*[3]+x*[4]+y*[5]+[6]",-9,9,-9,9);
    tf4 = new TF2("tf4","x^2*[0]+x*y*[1]+x*y*[2]+y^2*[3]+x*[4]+y*[5]+[6]",-9,9,-9,9);
	        
    TMatrix W1(2,2);
    TMatrix W2(2,2);

    TMatrix siginv1(2,2);
    TMatrix siginv2(2,2);

    TMatrix sigma1(2,2);
    TMatrix sigma2(2,2);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sigma1[i][j] = sig1[i][j];
            sigma2[i][j] = sig2[i][j];
        }
    }
    
    siginv1 = sigma1;
    siginv2 = sigma2;
    siginv1.Invert();
    siginv2.Invert();

    W1 = -0.5*siginv1;
    W2 = -0.5*siginv2;

    TMatrix w1(2,1);
    TMatrix w2(2,1);

    TMatrix mean1(2,1);
    TMatrix mean2(2,1);
    
    for (int i = 0; i < 2; i++) {
        mean1[i] = mu1[i];
        mean2[i] = mu2[i];
    }
    
    w1 = (siginv1*mean1);
    w2 = (siginv2*mean2);

    W1.Print();
    W2.Print();
    
    w1.Print();
    w2.Print();
    
    TMatrix m(1,1);
    
    TMatrix trans(1,2);
    
    //Calculate first parameter for class 1.
    trans.Transpose(mean1);
    m = trans*w1;
    
    double w10 = -0.5*(m[0][0])-0.5*log(sigma1.Determinant())+log(0.5);

    //Calculate first parameter for class 2.
    trans.Transpose(mean2);
    m = trans*w2;

    double w20 = -0.5*(m[0][0])-0.5*log(sigma2.Determinant())+log(0.5);

    //Set parameters into special variables
    double par10 = W1[0][0];
    double par11 = W1[0][1];
    double par12 = W1[1][0];
    double par13 = W1[1][1];
    double par14 = w1[0][0];
    double par15 = w1[1][0];
    double par16 = w10;

    double par20 = W2[0][0];
    double par21 = W2[0][1];
    double par22 = W2[1][0];
    double par23 = W2[1][1];
    double par24 = w2[0][0];
    double par25 = w2[1][0];
    double par26 = w20;

    tf3->SetParameters(par10,par11,par12,par13,par14,par15,par16);
    tf4->SetParameters(par20,par21,par22,par23,par24,par25,par26);

    //Error checking
    tf5 = new TF2("tf5", "tf3-tf4",-9,9,-9,9);
    tf5->SetTitle("The 2D Dichotomizer");
    tf5->Draw("colz");
    wrong = 0;
    
    for (int i = 0; i < SIZE; i++) {
        if (tf5->Eval(x1[i],y1[i]) < 0) {
            wrong++;
        }
        if (tf5->Eval(x2[i],y2[i]) > 0) {
            wrong++;
        }
    }
    
    error = (100.0*wrong)/(SIZE*2);

    printf("\nError for 2D: %d%%\n",error);
}
