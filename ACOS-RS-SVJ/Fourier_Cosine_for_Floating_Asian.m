format long;
warning off;
tic;
%% model parameters
%% Heston
% S_0=100;K=1;r=0.05;q=0;lambda=2.0;v_bar=0.01;eta=0.10;rho=0.50;v_0=0.01; %parameter of Heston model Gambaro 2020    q_cir=3
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates

% S_0=100;K=0.90;r=0.04;q=0;lambda=1;v_bar=0.09;eta=1;rho=-0.3;v_0=0.09;%parameter of Heston model corsaro 2018    q_cir=-0.82
% Tmat=1;N_monitor=50;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates

S_0=100;K=0.9;
r=0.0367;q=0;lambda=6.21;v_bar=0.019;eta=0.61;rho=-0.7;v_0=0.101^2; %parameter of Heston model fusai 2016   q_cir=-0.3658
Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;K=1;
% r=0.01;q=0.02;lambda=1.00;v_bar=0.01;eta=0.10;rho=-0.7;v_0=0.01; %parameter of Heston model Zeng 2016   q_cir=-0.3658
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor; 
%% Bates model
% K=0.9;
% S_0=100;
% r=0.0367;q=0;lambda=3.99;v_bar=0.014;eta=0.27;rho=-0.79;v_0=0.094^2;l_x=0.11;mu_x=-0.1391;sigma_x=0.15;%parameter of Bates model fusai 2016
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates
%% 3/2 model
% r=0.05;q=0;lambda=22.84;v_bar=0.4669^2;eta=8.56;rho=-0.99;v_0=0.060025;%parameter of 3/2 model 
% S_0=100;
% % K=0.9;% strike price
% Tmat=1;N_monitor=50;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates
%% 3/2 model+Kou
% r=0.05;q=0;lambda=22.84;v_bar=0.4669^2;eta=8.56;rho=-0.99;v_0=0.060025;l_x=5;p=0.4;eta_x1=10;eta_x2=5;%parameter of 3/2 model with DE jump
% S_0=100;
% % K=1.1;% strike price
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates
%% regime switching
% r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=0.3753;a2=-0.5503;b1=0.18;b2=0.6944;eta1=3.0465;eta2=3.0775;p1=0.3445;p2=0.3445;lambda1=5;lambda2=2; % J.L.Kirkby && regime switching && Kou-MN
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% S_0=100;K=0.9;
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;

% r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=-0.10;a2=-0.10;b1=0.30;b2=0.30;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching && Merton-Merton
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% S_0=100;K=0.9;% strike price
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;

% r=0.05;q=0;sigma1=0.10;sigma2=0.25;eta11=15;eta12=15;eta21=5;eta22=5;p1=0.35;p2=0.35;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching && Kou-Kou
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% S_0=100;K=1.10;% strike price
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor;
%% CGMY-CIR
% S_0=100;K=[0.90;1.00;1.10];r=0.05;q=0;C=4;G=50;M=60;Y=0.7;lambda=0.602;v_bar=1.2;eta=0.9;rho=0;v_0=1; %  CGMY+CIR
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;
% K=0.9;
% r=0.04;q=0;C=15.6840;G=10.2115;M=43.1510;Y=0.8;lambda=3.99;v_bar=0.014;eta=0.27;rho=0;v_0=0.008836; %  CGMY+CIR
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates
%% NIG-CIR
% S_0=100;
% % K=0.9;
% r=0.04;q=0;sigma_hat=0.84059;mu_hat=-11.00604;k_hat=0.00294;lambda=3.99;v_bar=0.014;eta=0.27;rho=0;v_0=0.008836;
% Tmat=1;N_monitor=250;delta_t=Tmat/N_monitor; %maturity and the number of monitor dates

% S_0=100;
% K=0.90;
% r=0.01;q=0.02;sigma_hat=0.1;mu_hat=-0.5;k_hat=0.01;lambda=1;v_bar=1;eta=1;rho=0;v_0=1; % zeng 2016 && NIG+CIR
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;
%% grid of state variable
N=600;      % the number of Fourier Cosine expansion terms
% c1=(r-q)*Tmat+(1-exp(-lambda*Tmat))*(v_bar-v_0)/(2*lambda)-1/2*v_bar*Tmat;     % 1st cumulant
% c2=1/(8*lambda^3)*(eta*Tmat*lambda*exp(-lambda*Tmat)*(v_0-v_bar)*(8*lambda*rho-4*eta)+lambda*rho*eta*(1-exp(-lambda*Tmat))*(16*v_bar-8*v_0)+2*v_bar*lambda*Tmat*(-4*lambda*rho*eta+eta^2+4*lambda^2)+...  
%     eta^2*((v_bar-2*v_0)*exp(-2*lambda*Tmat)+v_bar*(6*exp(-lambda*Tmat)-7)+2*v_0)+8*lambda^2*(v_0-v_bar)*(1-exp(-lambda*Tmat)));         % 2nd cumulant
% L=40;
% a=c1-L*sqrt(abs(c2));
% b=c1+L*sqrt(abs(c2)); %truncation boundary
L=5;
a=-L;b=L;
omega=(0:N-1)'*pi/(b-a);

Nx=800;
[nodesx,wx]=lgwt(Nx,a-log(1-exp(a)),b);
x=flipud(nodesx);
%% grid of variance
q_CIR=2*lambda*v_bar/eta^2-1;
zeta_CIR=2*lambda/((1-exp(-lambda*delta_t))*eta^2);   % zeta_CIR is large, as lambda*Delta is close to zero
zeta_CIR2=2*lambda/((1-exp(-lambda*Tmat))*eta^2);   % This part is for determing the truancation boudary for log-variance. It will be used in conditional density p(sigma_T) condition on sigma_ini.
mean_var=log(v_0*exp(-lambda*Tmat)+v_bar*(1-exp(-lambda*Tmat)));
a_v_ini=mean_var-5/(1+q_CIR);    % a proper initial guess for interval boundary
b_v_ini=mean_var+2/(1+q_CIR);
sigma_ini=log(v_0);
TOL=10^(-8);                     % error tolerance for conditional density function
eps=10^(-7);                     % Choose a proper error tolerance for finding early exercise boundary and boundary for log variance
a_v=NewtonIterate_low(a_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);    % interval boundary for log-variance. We use Tmat instead of Delta here.
b_v=NewtonIterate_up(b_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);
n_v=50;                         % total nodes for log-variance. When N=252, J=155 is enough.
low_var=exp(a_v);                % boundary for variance
up_var=exp(b_v);

[nodes,w]=lgwt(n_v,a_v,b_v);       % compute corresponding nodes and weights for Gaus-Legendre.
Zeta=fliplr(nodes.');  

%% grid of recipal of variance
% lambda1=lambda*v_bar;v_bar1=(lambda+eta^2)/(lambda*v_bar);eta1=-eta;
% v_ini=1/v_0;
% q_CIR=2*lambda1*v_bar1/eta1^2-1;
% zeta_CIR=2*lambda1/((1-exp(-lambda1*delta_t))*eta1^2);   % zeta_CIR is large, as lambda*Delta is close to zero
% zeta_CIR2=2*lambda1/((1-exp(-lambda1*Tmat))*eta1^2);
% mean_var=log(v_ini*exp(-lambda1*Tmat)+v_bar1*(1-exp(-lambda1*Tmat)));
% a_v_ini=mean_var-5/(1+q_CIR);    % a proper initial guess for interval boundary
% b_v_ini=mean_var+2/(1+q_CIR);
% sigma_ini=log(v_ini);
% TOL=10^(-7);                     % error tolerance for conditional density function
% eps=10^(-5);                     % Choose a proper error tolerance for finding early exercise boundary and boundary for log variance
% a_v=NewtonIterate_low(a_v_ini,zeta_CIR2,sigma_ini,lambda1,Tmat,q_CIR,TOL,eps);    % interval boundary for log-variance. We use Tmat instead of Delta here.
% b_v=NewtonIterate_up(b_v_ini,zeta_CIR2,sigma_ini,lambda1,Tmat,q_CIR,TOL,eps);
% n_v=55;                         % total nodes for log-variance. When N=252, J=155 is enough.
% low_var=exp(a_v);                % boundary for variance
% up_var=exp(b_v);
% 
% [nodes,w]=lgwt(n_v,a_v,b_v);       % compute corresponding nodes and weights for Gauss-Legendre.
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
omega_Chara=reshape(omega-1i,[],1,1);
Zeta_t=reshape(Zeta,1,[],1);
Zeta_s=reshape(Zeta,1,1,[]);
psi=sqrt(lambda^2-2*1i*omega_Chara.*(rho*lambda/eta-1/2+1i*omega_Chara*(1-rho^2)/2)*eta^2);
Bessel_chf_1=4*psi.*exp(-1/2*psi*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
Bessel_chf=Bessel_chf_1.*Bessel_chf_2;
Chara_1=exp(1/2*lambda*delta_t)*Bessel_chf_1/2;
Chara_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
Exp1=1i*omega_Chara*(r-q-rho*lambda*v_bar/eta)*delta_t;
Exp2=(-1i*omega_Chara*rho*eta+lambda)/eta^2;
Exp3=psi.*(1+exp(-psi*delta_t))./((1-exp(-psi*delta_t))*eta^2);
Zeta_m=exp(Zeta_t)-exp(Zeta_s);
Zeta_p=exp(Zeta_t)+exp(Zeta_s);
Chara=(exp(-(r-q)*delta_t)*Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));

Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));

Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
Charav0=(exp(-(r-q)*delta_t)*Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% Bates model
% omega_Chara=reshape(omega-1i,[],1,1);
% Zeta_t=reshape(Zeta,1,[],1);
% Zeta_s=reshape(Zeta,1,1,[]);
% psi=sqrt(lambda^2-2*1i*omega_Chara.*(rho*lambda/eta-1/2+1i*omega_Chara*(1-rho^2)/2)*eta^2);
% Bessel_chf_1=4*psi.*exp(-1/2*psi*delta_t)./(eta^2*(1-exp(-psi*delta_t)));
% Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
% Bessel_chf=Bessel_chf_1.*Bessel_chf_2;
% Chara_1=exp(1/2*lambda*delta_t)*Bessel_chf_1/2;
% Chara_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
% Exp1=1i*omega_Chara*(r-q-l_x*(exp(mu_x+sigma_x^2/2)-1)-rho*lambda*v_bar/eta)*delta_t+l_x*(exp(1i*omega_Chara*mu_x-sigma_x^2*omega_Chara.^2/2)-1)*delta_t;
% Exp2=(-1i*omega_Chara*rho*eta+lambda)/eta^2;
% Exp3=psi.*(1+exp(-psi*delta_t))./((1-exp(-psi*delta_t))*eta^2);
% Zeta_m=exp(Zeta_t)-exp(Zeta_s);
% Zeta_p=exp(Zeta_t)+exp(Zeta_s);
% Chara=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% 3/2 model
% v1=q_CIR;zeta1=zeta_CIR;
% nu=sqrt(v1^2-8i*(omega-1i)*(rho*lambda/eta+rho*eta/2-1/2)/eta^2+4*(omega-1i).^2*(1-rho^2)/eta^2);
% Bessel_var=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta'+Zeta));
% Chara_1=zeta1*exp(1i*(omega-1i)*(r-q-rho*lambda*v_bar/eta)*delta_t+Zeta+v1*lambda*v_bar/2*delta_t);
% Zeta_m=Zeta'-Zeta;
% Exp1=-1i*(omega-1i)*rho/eta+v1/2;
% Exp2=-zeta1*(exp(Zeta')+exp(Zeta-lambda*v_bar*delta_t));
% 
% w_diag=ones(N,n_v,n_v)+reshape(-0.5*eye(n_v,n_v),1,n_v,n_v);
% Bessel_chf1=zeros(N,n_v,n_v);
% Bessel_var_upper=triu(Bessel_var);
% Bessel_threshold=50;
% Index_Besselvarupper1=find(Bessel_var_upper>=Bessel_threshold);
% Index_Besselvarupper2=find(Bessel_var_upper~=0&Bessel_var_upper<Bessel_threshold);
% Besselvarupper_Matrix=zeros(N,1,1)+reshape(Bessel_var_upper,1,n_v,n_v);
% Index_Besselvarupper_Matrix1=find(Besselvarupper_Matrix>=Bessel_threshold);
% Index_Besselvarupper_Matrix2=find(Besselvarupper_Matrix~=0&Besselvarupper_Matrix<Bessel_threshold);
% Bessel_chf1(Index_Besselvarupper_Matrix2)=reshape(cbesseli(nu,Bessel_var_upper(Index_Besselvarupper2)),[],1);
% math('matlab2math','Nu',nu+zeros(1,length(Index_Besselvarupper1)));
% math('matlab2math','Besselvar',Bessel_var_upper(Index_Besselvarupper1).'+zeros(N,1));
% Bessel_chf1(Index_Besselvarupper_Matrix1)=math('math2matlab','Exp[-Besselvar]*BesselI[Nu,Besselvar]+0.I');
% Bessel_chf2=permute(Bessel_chf1,[1,3,2]);
% Bessel_chf=(Bessel_chf1+Bessel_chf2).*w_diag;
% Zeta_m=reshape(Zeta_m,1,n_v,n_v);
% Exp2=reshape(Exp2,1,n_v,n_v);
% Bessel_var=reshape(Bessel_var,1,n_v,n_v);
% Chara=(exp(-(r-q)*delta_t)*Chara_1.*exp(Exp1.*Zeta_m+Exp2+Bessel_var).*Bessel_chf);
% 
% Bessel_chf0=zeros(N,n_v);
% Bessel_var_0=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta-log(v_0)));
% Index_Besselvar01=find(Bessel_var_0>=Bessel_threshold);
% Index_Besselvar02=find(Bessel_var_0<Bessel_threshold);
% Bessel_chf0(:,Index_Besselvar02)=cbesseli(nu,Bessel_var_0(Index_Besselvar02));
% math('matlab2math','Nu0',nu+zeros(1,length(Index_Besselvar01)));
% math('matlab2math','Besselvar0',Bessel_var_0(Index_Besselvar01)+zeros(N,1));
% Bessel_chf0(:,Index_Besselvar01)=math('math2matlab','Exp[-Besselvar0]*BesselI[Nu0,Besselvar0]');
% 
% Exp1_0=(-1i*(omega-1i)*rho/eta+v1/2)*(Zeta+log(v_0));
% Exp2_0=-zeta1*(exp(Zeta)+exp(-lambda*v_bar*delta_t)/v_0);
% Charav0=(exp(-(r-q)*delta_t)*Chara_1.*exp(Exp1_0+Exp2_0+1*Bessel_var_0).*Bessel_chf0);



%% 3/2 model with DE jump
% v1=q_CIR;zeta1=zeta_CIR;
% nu=sqrt(v1^2-8i*(omega-1i)*(rho*lambda/eta+rho*eta/2-1/2)/eta^2+4*(omega-1i).^2*(1-rho^2)/eta^2);
% k_Kou=p*eta_x1/(eta_x1-1)+(1-p)*eta_x2/(eta_x2+1)-1;
% Bessel_var=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta'+Zeta));
% Chara_1=zeta1*exp(1i*(omega-1i)*(r-q-l_x*k_Kou-rho*lambda*v_bar/eta)*delta_t+l_x*(p*eta_x1./(eta_x1-1i*(omega-1i))+(1-p)*eta_x2./(eta_x2+1i*(omega-1i))-1)*delta_t+Zeta+v1*lambda*v_bar/2*delta_t);
% Zeta_m=Zeta'-Zeta;
% Exp1=-1i*(omega-1i)*rho/eta+v1/2;
% Exp2=-zeta1*(exp(Zeta')+exp(Zeta-lambda*v_bar*delta_t));
% 
% w_diag=ones(N,n_v,n_v)+reshape(-0.5*eye(n_v,n_v),1,n_v,n_v);
% Bessel_chf1=zeros(N,n_v,n_v);
% Bessel_var_upper=triu(Bessel_var);
% Bessel_threshold=60;
% Index_Besselvarupper1=find(Bessel_var_upper>=Bessel_threshold);
% Index_Besselvarupper2=find(Bessel_var_upper~=0&Bessel_var_upper<Bessel_threshold);
% Besselvarupper_Matrix=zeros(N,1,1)+reshape(Bessel_var_upper,1,n_v,n_v);
% Index_Besselvarupper_Matrix1=find(Besselvarupper_Matrix>=Bessel_threshold);
% Index_Besselvarupper_Matrix2=find(Besselvarupper_Matrix~=0&Besselvarupper_Matrix<Bessel_threshold);
% Bessel_chf1(Index_Besselvarupper_Matrix2)=reshape(cbesseli(nu,Bessel_var_upper(Index_Besselvarupper2)),[],1);
% math('matlab2math','Nu',nu+zeros(1,length(Index_Besselvarupper1)));
% math('matlab2math','Besselvar',Bessel_var_upper(Index_Besselvarupper1).'+zeros(N,1));
% Bessel_chf1(Index_Besselvarupper_Matrix1)=math('math2matlab','Exp[-Besselvar]*BesselI[Nu,Besselvar]+0.I');
% Bessel_chf2=permute(Bessel_chf1,[1,3,2]);
% Bessel_chf=(Bessel_chf1+Bessel_chf2).*w_diag;
% Zeta_m=reshape(Zeta_m,1,n_v,n_v);
% Exp2=reshape(Exp2,1,n_v,n_v);
% Bessel_var=reshape(Bessel_var,1,n_v,n_v);
% Chara=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*exp(Exp1.*Zeta_m+Exp2+Bessel_var).*Bessel_chf);
% 
% Bessel_chf0=zeros(N,n_v);
% Bessel_var_0=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta-log(v_0)));
% Index_Besselvar01=find(Bessel_var_0>=Bessel_threshold);
% Index_Besselvar02=find(Bessel_var_0<Bessel_threshold);
% Bessel_chf0(:,Index_Besselvar02)=cbesseli(nu,Bessel_var_0(Index_Besselvar02));
% math('matlab2math','Nu0',nu+zeros(1,length(Index_Besselvar01)));
% math('matlab2math','Besselvar0',Bessel_var_0(Index_Besselvar01)+zeros(N,1));
% Bessel_chf0(:,Index_Besselvar01)=math('math2matlab','Exp[-Besselvar0]*BesselI[Nu0,Besselvar0]');
% 
% Exp1_0=(-1i*(omega-1i)*rho/eta+v1/2)*(Zeta+log(v_0));
% Exp2_0=-zeta1*(exp(Zeta)+exp(-lambda*v_bar*delta_t)/v_0);
% Charav0=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*exp(Exp1_0+Exp2_0+1*Bessel_var_0).*Bessel_chf0);


%% regime switching model
%% MN-Kou
% mu1=r-q-sigma1^2/2-lambda1*(p1*eta1/(eta1-1)+(1-p1)*eta2/(eta2+1)-1);% double exponential
% mu2=r-q-sigma2^2/2-lambda2*(p2*exp(a1+1/2*b1^2)+(1-p2)*exp(a2+1/2*b2^2)-1);% mixed normal
% Psi1=1i*(omega-1i)*mu1-1/2*(omega-1i).^2*sigma1^2+lambda1*(p1*eta1./(eta1-1i*(omega-1i))+(1-p1)*eta2./(eta2+1i*(omega-1i))-1);% regime 1
% Psi2=1i*(omega-1i)*mu2-1/2*(omega-1i).^2*sigma2^2+lambda2*(p2*exp(1i*(omega-1i)*a1-1/2*(omega-1i).^2*b1^2)+(1-p2)*exp(1i*(omega-1i)*a2-1/2*(omega-1i).^2*b2^2)-1);% regime 2
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
% Chara=(exp(-(r-q)*delta_t)*permute(Epsilon,[3,2,1]));
% w=ones(2,1);

%% Merton-Merton 
% mu1=r-q-sigma1^2/2-lambda1*(exp(a1+1/2*b1^2)-1); % Merton
% mu2=r-q-sigma2^2/2-lambda2*(exp(a2+1/2*b2^2)-1); % Merton
% Psi1=1i*(omega-1i)*mu1-1/2*(omega-1i).^2*sigma1^2+lambda1*(exp(1i*(omega-1i)*a1-1/2*(omega-1i).^2*b1^2)-1);% regime 1
% Psi2=1i*(omega-1i)*mu2-1/2*(omega-1i).^2*sigma2^2+lambda2*(exp(1i*(omega-1i)*a2-1/2*(omega-1i).^2*b2^2)-1);% regime 2
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
% Chara=(exp(-(r-q)*delta_t)*permute(Epsilon,[3,2,1]));
% w=ones(2,1);

%% Kou-Kou
% mu1=r-q-sigma1^2/2-lambda1*(p1*eta11/(eta11-1)+(1-p1)*eta21/(eta21+1)-1);% double exponential
% mu2=r-q-sigma2^2/2-lambda2*(p2*eta12/(eta12-1)+(1-p2)*eta22/(eta22+1)-1);% double exponential
% Psi1=1i*(omega-1i)*mu1-1/2*(omega-1i).^2*sigma1^2+lambda1*(p1*eta11./(eta11-1i*(omega-1i))+(1-p1)*eta21./(eta21+1i*(omega-1i))-1);% regime 1
% Psi2=1i*(omega-1i)*mu2-1/2*(omega-1i).^2*sigma2^2+lambda2*(p2*eta12./(eta12-1i*(omega-1i))+(1-p2)*eta22./(eta22+1i*(omega-1i))-1);% regime 2
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
% Chara=(exp(-(r-q)*delta_t)*permute(Epsilon,[3,2,1]));
% w=ones(2,1);

%% CGMY-CIR
% omega_Chara=reshape(omega-1i,[],1,1);
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
% Chara=(exp(-(r-q)*delta_t)*Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=(exp(-(r-q)*delta_t)*Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% NIG-CIR
% omega_Chara=reshape(omega-1i,[],1,1);
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
% Chara=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*Chara_2.*exp(Exp1-Exp2.*Zeta_m-Exp3.*Zeta_p+abs(real(Bessel_chf))).*besseli(q_CIR,Bessel_chf,1));
% 
% Bessel_chf_v0_2=exp(Zeta_t/2)*sqrt(v_0);
% Chara_v0_2=exp(Zeta_t+q_CIR/2*(Zeta_t-(log(v_0)-lambda*delta_t)));
% 
% Bessel_chf_v0=Bessel_chf_1.*Bessel_chf_v0_2;
% Charav0=gpuArray(exp(-(r-q)*delta_t)*Chara_1.*Chara_v0_2.*exp(Exp1-Exp2.*(exp(Zeta_t)-v_0)-Exp3.*(exp(Zeta_t)+v_0)+abs(real(Bessel_chf_v0))).*besseli(q_CIR,Bessel_chf_v0,1));

%% Fourier Cosine expansion at first time step


VN=2/(b-a)*(-(K*(N_monitor+1)-1)./omega.*sin(omega*(-log(K*(N_monitor+1)-1)-a))-((K*(N_monitor+1)-1)*cos(omega*(-log(K*(N_monitor+1)-1)-a))-...
    (K*(N_monitor+1)-1)*omega.*sin(omega*(-log(K*(N_monitor+1)-1)-a))-(-1).^(0:N-1)'.*exp(-b))./(1+omega.^2));
VN(1)=2*((K*(N_monitor+1)-1)*(b+log(K*(N_monitor+1)-1)-1)+exp(-b))/(b-a);
BetaN_sum=sum(VN*w.'.*Chara,2);
BetaN=reshape(BetaN_sum,N,n_v); % N*n_v dimension

% t2=toc;
% tic;

%% backward induction
Betak=(BetaN);
ww=[1/2,ones(1,N-1)];
for tk=1:N_monitor-2
    Vk=real(ww.*Betak.'*MF);
    Betak_sum=sum(w.'.*Vk.'.*Chara,2);
    Betak=reshape(Betak_sum,N,n_v);
end

%% recovery of option price
%% SV & TCL

V0=real(ww.*Betak.'*MF);
Beta0_sum=sum(w.'.*V0.'.*Charav0,2);
Beta0=reshape(Beta0_sum,1,N);

V_c=(real(S_0*exp(-q*Tmat)/(N_monitor+1)*ww.*Beta0*exp(-a*1i*omega)));
V_p=V_c-exp(-r*Tmat)*S_0*(K*exp((r-q)*Tmat)-1/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t)));


% V_p_Heston=[V_p_Heston;log(abs(V_p_benchmark_Heston-V_p))];

% gpuArray
% gather
%% regime switching levy

% V0=real(ww.*Betak.'*MF);
% Beta0_sum_1=sum(w.'.*V0.'.*Chara(:,:,1),2);
% Beta0_1=reshape(Beta0_sum_1,1,N);
% Beta0_sum_2=sum(w.'.*V0.'.*Chara(:,:,2),2);
% Beta0_2=reshape(Beta0_sum_2,1,N);
% 
% V_c_1=(real(S_0*exp(-q*Tmat)/(N_monitor+1)*ww.*Beta0_1*exp(-a*1i*omega)));
% V_c_2=(real(S_0*exp(-q*Tmat)/(N_monitor+1)*ww.*Beta0_2*exp(-a*1i*omega)));
% V_p_1=V_c_1-exp(-r*Tmat)*S_0*(K*exp((r-q)*Tmat)-1/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t)));
% V_p_2=V_c_2-exp(-r*Tmat)*S_0*(K*exp((r-q)*Tmat)-1/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t)));

% V_p_RSL2=[V_p_RSL2;log(abs(V_p_benchmark_RSL2-V_p_1))];

% plot(100:100:400,V_p_RSL1,'-*');
% hold on;
% plot(200:100:500,V_p_RSL2,'-o');
% hold on;
% plot(20:10:50,V_p_Heston,'-+');
% hold on;
% plot(20:10:50,V_p_Bates,'-x');
% hold on;
% plot(20:10:50,V_p_3o2,'-s');
% hold on;
% plot(15:10:45,V_p_3o2DE,'-d');
% hold on;
% plot(25:10:55,V_p_NIGCIR,'-p');
% hold on;
% plot(25:10:55,V_p_CGMYCIR,'-h');
% hold off;
% xlabel('J');
% ylabel('ln(\epsilon(V_0))');
% axis padded
% title('Convergence rate w.r.t. J');



% t=toc;
% 
% % k=1;avertime=0;avertime2=0;
% avertime=(avertime*(k-1)+t)/k;
% % avertime2=(avertime2*(k-1)+t2)/k;
% k=k+1; 

toc;