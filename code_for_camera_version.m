%% constructing a piecewise flat distribution, for NIPS rebuttal


% pieice-wise flat distribution

%f = @(x)((x<0.5 & x>-0.5).*0.5 + (x>-1.5 & x<=-0.5)./8 + (x>=0.5 & x<1.5)./8 + (x<=-1.5 & x>-3.5)./32 + (x>=1.5 & x<3.5)./32 ...
%+ (x<=-3.5 & x>-7.5)./128 + (x>=3.5 & x<7.5)./128 + (x<=-7.5 & x>-15.5)./512 + (x>=7.5 & x<15.5)./512);

% Weilbull distribution with exponent k = 0.5

h = 2;
f = @(x) (exp(-abs(x).^(1/2*h)));

normalizing_constant = h*gamma(2*h);
f = @(x) (f(x)./normalizing_constant);

betastar = 1;
density = @(x) (0.5.*f(x-betastar)+0.5.*f(x+betastar));
ratio_t = @(x,beta) ((f(x-beta)-f(x+beta))./(f(x-beta)+f(x+beta)));

clf;

beta0s = [0.1,0.2,0.3,0.4,0.5] ;

for beta0 = beta0s

    T = 10;
    
    iters = zeros(1,T);
    
    beta = beta0;
    
    for t=1:T
        integrand = @(x) (x.*ratio_t(x,beta).*density(x));
        iters(t) = integral(integrand,-10,10);
        beta = iters(t);
    end
    
    plot(0:T,[beta0, iters], '.-', 'linewidth', 1, 'markersize', 8); hold on;

end

hold off;
xlabel('iteration')
ylabel('LS-EM iterates \beta_t')
ylim([0,0.5])
h = legend('\beta_0 = 0.1', '\beta_0 = 0.2','\beta_0 = 0.3','\beta_0 = 0.4','\beta_0 = 0.5',...
    'location', 'none');
set(h, 'position', [0.7050    0.5801    0.1875    0.3182]);
set(gca, 'fontsize', 12);
set(h, 'fontsize', 8);

%export_fig('../../NIPS2019_initial_submission/non_convergence_new', '-pdf', '-transparent', gcf);





%% numerical gradient with respect to z for the function M(z,beta)
h = 2;
beta = 0.1;
betastar = 1;
n = 20;
zs = linspace(beta,betastar,n);

N = 10000000;
x = (gamrnd(2*h,1,N,1).^(2*h)).*(2.*(binornd(1,0.5,N,1)-0.5));


s = @(x) (x./abs(x));
f = @(x) ((abs(x+beta)).^(1/(2*h)) - (abs(x-beta)).^(1/(2*h)));
f_prime = @(x)((1/(2*h))*s(x+beta).*(abs(x+beta)).^(1/(2*h)-1)-(1/(2*h)).*s(x-beta).*(abs(x-beta)).^(1/(2*h)-1));
tanh_prime= @(x) (4./(exp(x)+exp(-x)).^2);

val1s = zeros(n,1);
val2s = zeros(n,1);
for i = 1:n
    z = zs(i);
    xz = x+z;
    val1s(i) = sum(tanh(0.5.*f(xz)))/N;
    val2s(i) = sum(0.5.*xz.*f_prime(xz).*tanh_prime(0.5.*f(xz)))/N;
end

hold on;

plot(zs,val1s,'b');
plot(zs,val2s,'r');


%% LS-EM step is approximately EM step

M = 10;
exponent = 1;
density = @(x) (exp(-abs(x).^exponent));
C = integral(density,-M,M);
density = @(x) (1/C.*exp(-abs(x).^exponent));
g = @(x) (abs(x).^exponent);
betastar =1;
betas = [0.1,0.5,0.8,1.2,1.5];
for j = 1:length(betas)
beta = betas(j);
tbeta = 0:0.1:2;
Q_vals = zeros(length(tbeta),1);
p1 = @(x) (density(x-beta)./(density(x-beta)+density(x+beta)));
p2 = @(x) (1-p1(x));
integrand = @(x,b) (p1(x).*g(x-b)+p2(x).*g(x+b));
for i = 1:length(tbeta)
    
    Q_vals(i) = integral(@(x) (density(x).*integrand(x+betastar,tbeta(i))),-M,M);
end

beta_plus_func = @(x) (x.*tanh(0.5.*(g(x+beta)-g(x-beta))));
beta_plus = integral(@(x) (density(x).*beta_plus_func(x+betastar)),-M,M);
val_at_beta = integral(@(x) (density(x).*integrand(x+betastar,beta)),-M,M);
val_at_beta_plus = integral(@(x) (density(x).*integrand(x+betastar,beta_plus)),-M,M);

hold on;
plot(tbeta,Q_vals);
scatter(beta,val_at_beta,'b');
scatter(beta_plus,val_at_beta_plus,'r');
end
    

%% Robustness

M = 10;
true_exponent = 3;
fit_exponents = [1,1.5,2,3];
betastar = 1;
betas = 0:0.1:2*betastar;
true_density = @(x) (exp(-abs(x).^(true_exponent)));
C = integral(true_density,-M,M);
true_density = @(x) (1./C.*exp(-abs(x).^(true_exponent)));
var = integral(@(x) (x.^2.*true_density(x)),-M,M);
true_density = @(x) (1./C./sqrt(var).*exp(-abs(x./sqrt(var)).^(true_exponent)));

for j  = 1:length(fit_exponents)
exponent = fit_exponents(j);
iters = zeros(length(betas),1);
fit_density = @(x)(exp(-abs(x).^(exponent)));
C =  integral(fit_density,-M,M);
var  = integral(@(x) (1./C.*x.^2.*fit_density(x)),-M,M); 
for i = 1:length(betas)
    beta = betas(i);
    integrand = @(x) (x.*tanh(0.5.*(abs((x+beta)./sqrt(var)).^exponent - abs((x-beta)./sqrt(var)).^exponent)));
    iters(i) = integral(@(x) (true_density(x-betastar).*integrand(x)),-M,M);
end
hold on;
plot(betas,iters);
end

hold on;
plot(betas,betas);

%% Spurious fixed point along the orthogonal direction 


exponent = 2.2;
M = 10;
density = @(x,y) exp(-(x.^2+y.^2).^exponent);
constant = integral2(density,-M,M,-M,M);
density = @(x,y) (density(x,y)./constant);

betastar = [1,0];
beta = [0,0.1];

T = 5;
iter1 = zeros(T+1,1);
iter2 = zeros(T+1,1);

iter1(1) = beta(1);
iter2(1) = beta(2);

for t = 2:T+1
    ratio = @(x,y) (0.5.*(((x+beta(1)).^2+(y+beta(2)).^2).^exponent - ((x-beta(1)).^2+(y-beta(2)).^2).^exponent));
    integrand1 = @(x,y) (density(x-betastar(1),y-betastar(2)).*x.*tanh(0.5.*ratio(x,y)));
    integrand2 = @(x,y) (density(x-betastar(1),y-betastar(2)).*y.*tanh(0.5.*ratio(x,y)));
    iter1(t) = integral2(integrand1,-M,M,-M,M);
    iter2(t) = integral2(integrand2,-M,M,-M,M);
    beta = [iter1(t),iter2(t)];
end

hold on;
norm_diff = sum(([iter1,iter2]-betastar).^2,2);

plot(1:T+1,norm_diff);

    
    








