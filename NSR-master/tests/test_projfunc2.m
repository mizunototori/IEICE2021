nn = 1;
N = 10;

s = [0.498117443123917,-0.250147745967313,0.120701165599073,-0.0733167842633556,0.363149925925690,-0.353954844241657,0.0105814433636021,0.179575821726580,0.357707364028597,0.501881554130832]';
k1 = sqrt(N) - (sqrt(N) - 1) * 0.1;

[x, usediters] = projfunc(s, k1, 1, nn);

disp(x);
