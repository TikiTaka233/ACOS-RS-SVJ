format long;
warning off;
tic;
%% model parameters
%% Heston
% S_0=100;K=1;r=0.05;q=0;lambda=2.0;v_bar=0.01;eta=0.10;rho=0.50;v_0=0.01; %parameter of Heston model Gambaro 2020    q_cir=3
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates

% S_0=100;K=1;r=0.0367;q=0;lambda=6.21;v_bar=0.019;eta=0.61;rho=-0.7;v_0=0.101^2; %parameter of Heston model fusai 2016   q_cir=-0.3658
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;
% % K=0.9;
% r=0.04;q=0;lambda=1;v_bar=0.09;eta=1;rho=-0.3;v_0=0.09;%parameter of Heston model corsaro 2018    q_cir=-0.82
% Tmat=1;N_monitor=50;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates

%% regime switching
% r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=0.3753;a2=-0.5503;b1=0.18;b2=0.6944;eta1=3.0465;eta2=3.0775;p1=0.3445;p2=0.3445;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching && Mixed normal and Double exponential
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% S_0=100;K=1;
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;

% r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=-0.10;a2=-0.10;b1=0.30;b2=0.30;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% S_0=100;K=1;% strike price
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;

r=0.05;q=0;sigma1=0.10;sigma2=0.25;eta11=15;eta12=15;eta21=5;eta22=5;p1=0.35;p2=0.35;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching && Kou-Kou
Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
S_0=100;
% K=0.9;% strike price
Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;
%% CGMY-CIR
% S_0=100;K=1;r=0.05;q=0;C=4;G=50;M=60;Y=0.7;lambda=0.602;v_bar=1.2;eta=0.9;rho=0;v_0=1; %  CGMY+CIR
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;K=1;r=0.04;q=0;C=15.6840;G=10.2115;M=43.1510;Y=0.8;lambda=3.99;v_bar=0.014;eta=0.27;rho=0;v_0=0.008836; %  CGMY+CIR
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

%% NIG-CIR
% S_0=100;
% % K=0.9;
% r=0.04;q=0;sigma_hat=0.84059;mu_hat=-11.00604;k_hat=0.00294;lambda=3.99;v_bar=0.014;eta=0.27;rho=0;v_0=0.008836;
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;
% K=0.9;
% r=0.01;q=0.02;sigma_hat=0.1;mu_hat=-0.5;k_hat=0.01;lambda=1;v_bar=1;eta=1;rho=0;v_0=1; % zeng 2016 && NIG+CIR
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;
%% grid of state variable
N=140;      % the number of Fourier Cosine expansion terms
% c1=(r-q)*Tmat+(1-exp(-lambda*Tmat))*(v_bar-v_0)/(2*lambda)-1/2*v_bar*Tmat;     % 1st cumulant
% c2=1/(8*lambda^3)*(eta*Tmat*lambda*exp(-lambda*Tmat)*(v_0-v_bar)*(8*lambda*rho-4*eta)+lambda*rho*eta*(1-exp(-lambda*Tmat))*(16*v_bar-8*v_0)+2*v_bar*lambda*Tmat*(-4*lambda*rho*eta+eta^2+4*lambda^2)+...  
%     eta^2*((v_bar-2*v_0)*exp(-2*lambda*Tmat)+v_bar*(6*exp(-lambda*Tmat)-7)+2*v_0)+8*lambda^2*(v_0-v_bar)*(1-exp(-lambda*Tmat)));         % 2nd cumulant
% L=40;
% a=c1-L*sqrt(abs(c2));
% b=c1+L*sqrt(abs(c2)); %truncation boundary
a=-8;b=8;
omega=(0:N-1)'*pi/(b-a);

Nx=190;
[nodesx,wx]=lgwt(Nx,a,b);
x=flipud(nodesx);
%% grid of variance
% q_CIR=2*lambda*v_bar/eta^2-1;
% zeta_CIR=2*lambda/((1-exp(-lambda*delta_t))*eta^2);   % zeta_CIR is large, as lambda*Delta is close to zero
% zeta_CIR2=2*lambda/((1-exp(-lambda*Tmat))*eta^2);   % This part is for determing the truancation boudary for log-variance. It will be used in conditional density p(sigma_T) condition on sigma_ini.
% mean_var=log(v_0*exp(-lambda*Tmat)+v_bar*(1-exp(-lambda*Tmat)));
% a_v_ini=mean_var-5/(1+q_CIR);    % a proper initial guess for interval boundary
% b_v_ini=mean_var+2/(1+q_CIR);
% sigma_ini=log(v_0);
% TOL=10^(-8);                     % error tolerance for conditional density function
% eps=10^(-7);                     % Choose a proper error tolerance for finding early exercise boundary and boundary for log variance
% a_v=NewtonIterate_low(a_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);    % interval boundary for log-variance. We use Tmat instead of Delta here.
% b_v=NewtonIterate_up(b_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);
% n_v=210;                         % total nodes for log-variance. When N=252, J=155 is enough.
% low_var=exp(a_v);                % boundary for variance
% up_var=exp(b_v);
% 
% [nodes,w]=lgwt(n_v,a_v,b_v);       % compute corresponding nodes and weights for Guass-Legendre.
% Zeta=fliplr(nodes.');  

%% initialization

x=reshape(x,1,1,[]);
wx=reshape(wx,1,1,[]);
Matrix=2/(b-a)*wx.*(1+exp(x)).^(-1i*omega).*exp(1i*(x-a).*omega).*cos((x-a).*omega.');
MF=sum(Matrix,3);



%% hypergeometric function
% lnp=(0:N-1).'+(0:N-1);
% lnm=(0:N-1).'-(0:N-1);
% MF=-1i*(a-b)./(2*lnm.*lnp*pi).*(lnp.*hypergeom([-1i*lnm*pi/(a-b),-1+1i*omega*ones(1,N)],1-1i*lnm*pi/(a-b),-exp(a))+lnp.*hypergeom([-1i*lnp*pi/(a-b),-1+1i*omega*ones(1,N)],1-1i*lnp*pi/(a-b),-exp(a)))+...
%     1i*(a-b)./(2*lnm.*lnp*pi).*(exp(1i*lnm*pi).*lnp.*hypergeom([-1i*lnm*pi/(a-b),-1+1i*omega*ones(1,N)],1-1i*lnm*pi/(a-b),-exp(b))+exp(1i*lnp*pi).*lnm.*hypergeom([-1i*lnp*pi/(a-b),-1+1i*omega*ones(1,N)],1-1i*lnp*pi/(a-b),-exp(b)));
% 
% Matrix=2/(b-a)*(1+exp(x.')).^(1-1i*omega).*exp(1i*(x.'-a).*omega).*cos((x.'-a).*omega)*wx;
% MF(logical(eye(size(MF))))=Matrix;

    

%% conditional characteristic function
%% Heston model
% omega_Chara=reshape(omega,[],1,1);
% Zeta_t=reshape(Zeta,1,[],1);
% Zeta_s=reshape(Zeta,1,1,[]);
% psi=sqrt(lambda^2-2*1i*omega_Chara.*(rho*lambda/eta-1/2+1i*omega_Chara*(1-rho^2)/2)*eta^2);
% Bessel_chf_1=4*psi.*exp(-1/2*psi*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
% Bessel_chf=Bessel_chf_1.*Bessel_chf_2;
% Chara_1=exp(1/2*lambda*delta_t)*Bessel_chf_1/2;
% Chara_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
% Exp1=1i*omega_Chara*(r-q-rho*lambda*v_bar/eta)*delta_t;
% Exp2=(-1i*omega_Chara*rho*eta+lambda)/eta^2;
% Exp3=psi.*(1+exp(-psi*delta_t))./((1-exp(-psi*delta_t))*eta^2);
% Zeta_m=exp(Zeta_t)-exp(Zeta_s);
% Zeta_p=exp(Zeta_t)+exp(Zeta_s);
% Chara=(Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=(Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));


%% regime switching model
% mu1=r-q-sigma1^2/2-lambda1*(p1*eta1/(eta1-1)+(1-p1)*eta2/(eta2+1)-1);% double exponential
% mu2=r-q-sigma2^2/2-lambda2*(p2*exp(a1+1/2*b1^2)+(1-p2)*exp(a2+1/2*b2^2)-1);% mixed normal
% Psi1=1i*omega*mu1-1/2*omega.^2*sigma1^2+lambda1*(p1*eta1./(eta1-1i*omega)+(1-p1)*eta2./(eta2+1i*omega)-1);% regime 1
% Psi2=1i*omega*mu2-1/2*omega.^2*sigma2^2+lambda2*(p2*exp(1i*omega*a1-1/2*omega.^2*b1^2)+(1-p2)*exp(1i*omega*a2-1/2*omega.^2*b2^2)-1);% regime 2
% Epsilon_diag=zeros(2,2,N);
% Epsilon_diag(1,1,:)=Psi1;Epsilon_diag(2,2,:)=Psi2;
% 
% Epsilon_diag=Epsilon_diag+Q';
% Eig_1=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)-sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
%     2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
% Eig_2=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)+sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
%     2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
% Epsilon=exp(Eig_1*delta_t).*(eye(2,2)+zeros(2,2,N))+(exp(Eig_1*delta_t)-exp(Eig_2*delta_t))./(Eig_1-Eig_2).*(Epsilon_diag-Eig_1.*(eye(2,2)+zeros(2,2,N)));
% 
% n_v=2;% the number of state
% Chara=zeros(N,2,2,'gpuArray');
% Chara(:,1,1)=Epsilon(1,1,:);
% Chara(:,2,1)=Epsilon(1,2,:);
% Chara(:,1,2)=Epsilon(2,1,:);
% Chara(:,2,2)=Epsilon(2,2,:);
% w=ones(2,1);

%% Merton-Merton
% mu1=r-q-sigma1^2/2-lambda1*(exp(a1+1/2*b1^2)-1); % Merton
% mu2=r-q-sigma2^2/2-lambda2*(exp(a2+1/2*b2^2)-1); % Merton
% Psi1=1i*omega*mu1-1/2*omega.^2*sigma1^2+lambda1*(exp(1i*omega*a1-1/2*omega.^2*b1^2)-1);% regime 1
% Psi2=1i*omega*mu2-1/2*omega.^2*sigma2^2+lambda2*(exp(1i*omega*a2-1/2*omega.^2*b2^2)-1);% regime 2
% Epsilon_diag=zeros(2,2,N);
% Epsilon_diag(1,1,:)=Psi1;Epsilon_diag(2,2,:)=Psi2;
% 
% Epsilon_diag=Epsilon_diag+Q';
% Eig_1=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)-sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
%     2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
% Eig_2=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)+sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
%     2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
% Epsilon=exp(Eig_1*delta_t).*(eye(2,2)+zeros(2,2,N))+(exp(Eig_1*delta_t)-exp(Eig_2*delta_t))./(Eig_1-Eig_2).*(Epsilon_diag-Eig_1.*(eye(2,2)+zeros(2,2,N)));
% 
% n_v=2;% the number of state
% Chara=zeros(N,2,2,'gpuArray');
% Chara(:,1,1)=Epsilon(1,1,:);
% Chara(:,2,1)=Epsilon(1,2,:);
% Chara(:,1,2)=Epsilon(2,1,:);
% Chara(:,2,2)=Epsilon(2,2,:);
% w=ones(2,1);

%% Kou-Kou
mu1=r-q-sigma1^2/2-lambda1*(p1*eta11/(eta11-1)+(1-p1)*eta21/(eta21+1)-1);% double exponential
mu2=r-q-sigma2^2/2-lambda2*(p2*eta12/(eta12-1)+(1-p2)*eta22/(eta22+1)-1);% double exponential
Psi1=1i*omega*mu1-1/2*omega.^2*sigma1^2+lambda1*(p1*eta11./(eta11-1i*omega)+(1-p1)*eta21./(eta21+1i*omega)-1);% regime 1
Psi2=1i*omega*mu2-1/2*omega.^2*sigma2^2+lambda2*(p2*eta12./(eta12-1i*omega)+(1-p2)*eta22./(eta22+1i*omega)-1);% regime 2
Epsilon_diag=zeros(2,2,N);
Epsilon_diag(1,1,:)=Psi1;Epsilon_diag(2,2,:)=Psi2;

Epsilon_diag=Epsilon_diag+Q';
Eig_1=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)-sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
    2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
Eig_2=1/2*(Epsilon_diag(1,1,:)+Epsilon_diag(2,2,:)+sqrt(Epsilon_diag(1,1,:).^2+Epsilon_diag(2,2,:).^2-...
    2*Epsilon_diag(1,1,:).*Epsilon_diag(2,2,:)+4*Epsilon_diag(1,2,:).*Epsilon_diag(2,1,:)));
Epsilon=exp(Eig_1*delta_t).*(eye(2,2)+zeros(2,2,N))+(exp(Eig_1*delta_t)-exp(Eig_2*delta_t))./(Eig_1-Eig_2).*(Epsilon_diag-Eig_1.*(eye(2,2)+zeros(2,2,N)));

n_v=2;% the number of state
Chara=permute(Epsilon,[3,2,1]);
w=ones(2,1);
%% CGMY-CIR
% omega_Chara=reshape(omega,[],1,1);
% Zeta_t=reshape(Zeta,1,[],1);
% Zeta_s=reshape(Zeta,1,1,[]);
% psi=sqrt(lambda^2+2*eta^2*C*gamma(-Y)*((M^Y-(M-1i*omega_Chara).^Y+G^Y-(G+1i*omega_Chara).^Y)-1i*omega_Chara*(M^Y-(M-1)^Y+G^Y-(G+1)^Y)));
% Bessel_chf_1=4*psi.*exp(-1/2*psi*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
% Chara_1=2*psi.*exp(-1/2*(psi-lambda)*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Chara_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
% Exp1=1i*omega_Chara*(r-q)*delta_t;
% Exp2=(-1i*omega_Chara*rho*eta+lambda)/eta^2;
% Exp3=psi.*(1+exp(-psi*delta_t))./((1-exp(-psi*delta_t))*eta^2);
% Zeta_m=exp(Zeta_t)-exp(Zeta_s);
% Zeta_p=exp(Zeta_t)+exp(Zeta_s);
% 
% Bessel_chf=Bessel_chf_1.*Bessel_chf_2;
% Chara=(Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=(Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% NIG-CIR
% omega_Chara=reshape(omega,[],1,1);
% Zeta_t=reshape(Zeta,1,[],1);
% Zeta_s=reshape(Zeta,1,1,[]);
% psi=sqrt(lambda^2+2*eta^2/k_hat*(sqrt(1+k_hat*sigma_hat^2*omega_Chara.^2-2*1i*mu_hat*k_hat*omega_Chara)-1-1i*omega_Chara*(sqrt(1-k_hat*sigma_hat^2-2*mu_hat*k_hat)-1)));
% Bessel_chf_1=4*psi.*exp(-1/2*psi*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
% Chara_1=2*psi.*exp(-1/2*(psi-lambda)*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Chara_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
% Exp1=1i*omega_Chara*(r-q)*delta_t;
% Exp2=(-1i*omega_Chara*rho*eta+lambda)/eta^2;
% Exp3=psi.*(1+exp(-psi*delta_t))./((1-exp(-psi*delta_t))*eta^2);
% Zeta_m=exp(Zeta_t)-exp(Zeta_s);
% Zeta_p=exp(Zeta_t)+exp(Zeta_s);
% 
% Bessel_chf=Bessel_chf_1.*Bessel_chf_2;
% Chara=(Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=(Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% Fourier Cosine expansion at first time step

Nx_V=30;
[nodex_N,wx_N]=lgwt(Nx_V,a,log(K/(N_monitor+1-K)));
x_N=fliplr(nodex_N.');
VN=2/(b-a)*((K/(N_monitor+1)-exp(x_N)./(1+exp(x_N))).*cos(omega.*(x_N-a)))*wx_N;
BetaN_sum=sum(VN*w.'.*Chara,2);
BetaN=reshape(BetaN_sum,N,n_v); % N*n_v dimension



%% backward induction
Betak=BetaN;
ww=[1/2,ones(1,N-1)];
for tk=1:N_monitor-2
    Vk=real(ww.*Betak.'*MF);
    Betak_sum=sum(w.'.*Vk.'.*Chara,2);
    Betak=reshape(Betak_sum,N,n_v);
end

%% recovery of option price
%% Heston & CGMY-CIR

% V0=real(ww.*Betak.'*MF);
% Beta0_sum=sum(w.'.*V0.'.*Charav0,2);
% Beta0=reshape(Beta0_sum,1,N);
% 
% V_p=(real(exp(-r*Tmat)*(N_monitor+1)*ww.*Beta0*exp(-a*1i*omega)));

%% regime switching

V0=real(ww.*Betak.'*MF);
Beta0_sum_1=sum(w.'.*V0.'.*Chara(:,:,1),2);
Beta0_1=reshape(Beta0_sum_1,1,N);
% Beta0_sum_2=sum(w.'.*V0.'.*Chara(:,:,2),2);
% Beta0_2=reshape(Beta0_sum_2,1,N);

V_p_1=(real(exp(-r*Tmat)*(N_monitor+1)*ww.*Beta0_1*exp(-a*1i*omega)));
% V_p_2=(real(exp(-r*Tmat)*(N_monitor+1)*ww.*Beta0_2*exp(-a*1i*omega)));


t=toc;

% k=1;avertime=0;avertime2=0;
avertime=(avertime*(k-1)+t)/k;
% avertime2=(avertime2*(k-1)+t2)/k;
k=k+1; 


% toc;