//Formulas
//Pattern Rec Spring 2014
//Travis Cramer

void formula() {
	//Simple 1D formula
	f1 = new TF1("f1","sin(x)/x^2",0,10);
	f1->SetLineColor(kRed-6);

	//Formula with parameters
	f2 = new TF1("f2","[0]*exp(-0.5 * (x-[1])^2 / [2]^2)", 0, 2);

	f2->SetParNames("Amp","Mean","Sigma");
	f2->SetParameters(10, 3, 1);

	//a 2D formula
	f3 = new TF2("f3", "sin(x)*y + y/x",0,10,0,10);

	//Draw
	//Define canvas
	c1 = new TCanvas("c1","My Awesome Gaussian", 1000, 400);
	c1->Divide(3,1);

	c1->cd(1);
	f1->Draw();

	c1->cd(2);
	f2->Draw();

	c1->cd(3);
	gStyle->SetPalette(1);
	f3->Draw("surf1");
}