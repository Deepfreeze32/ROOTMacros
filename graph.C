//Graph
//Pattern Rec Spring 2014
//Travis Cramer

void graph() {
	tg = new TGraph("data.txt");
	tg->SetMarkerStyle(34);
	tg->SetMarkerColor(kRed);
	tg->Draw("AP");
}