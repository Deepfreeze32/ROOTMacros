void Cramer_mp1() {
	float x1[] = {9, 10, 10.7, 11, 11.5, 12, 13, 14, 15}; 
	float y1[] = {0, 0.08, 0.24, .21, 0.15, .18, .30, .17, .05}; 
 	float y2[] = {0.03, .14, .27, .33, .36, .3, .18, .04, 0}; 
	
 	c1 = new TCanvas("c1", "My Plots", 800, 400);
 	c1->Divide(2,1);

	tg1 = new TGraph(9,x1,y1);
	tg1->SetMarkerColor(kRed);
	tg1->SetMarkerStyle(20);
	tg2 = new TGraph(9,x1,y2);
	tg2->SetMarkerColor(kBlack);
	tg2->SetMarkerStyle(20);

	tf1 = new TF1("tf1","gaus(0) + gaus(3)", 9, 15);

	c1->cd(1);
	tg1->Draw("AP");
	tf1->SetLineColor(kRed);
	//tf1->Draw();
	tg1->Fit(tf1);

	c1->cd(2);
	tg2->Draw("AP");
	tf1->SetLineColor(kBlack);
	tg2->Fit(tf1);
}
