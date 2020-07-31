% This script demonstrates the 1D test case in 
% Brain Dynamics and Structure-Function Relationships via Spectral Factorization and the Transfer Function
% J. A. Henderson, M. Dhamala and P. A. Robinson, 2020
%
% Created by J. A. Henderson, July 2020

clear all

% Generate 2D grid 
Npts=7; % number of points in each dimension(odd)
circum=pi*15e-2; % circumference in m
rad=circum/(2*pi); % circle radius (periodic boundaries) 
dtheta=2*pi/Npts; % angular resolution
dX=rad*dtheta; % spatial resolution
theta=0:dtheta:2*pi-dtheta; % anglular coordinate of each point

for i=1:Npts
    for j=1:Npts
        sqanglesX(i,j)=theta(i);
        sqanglesY(i,j)=theta(j);       
    end
end

anglesX=reshape(sqanglesX,1,Npts^2);
anglesY=reshape(sqanglesY,1,Npts^2);

G=ones(Npts^2,Npts^2); % set gains to one

% Neural Field Theory Parameters
alpha=100; % synaptic decay rate [s^-1]
beta=350; % synaptic rise rate [s^-1]
r_e=84e-3; % axonal range [m]
vel=9; % conduction speed [m/s]
gamma_e=vel/r_e; % decay rate [s^-1]


fmax=0.5*vel/(circum/Npts); % maximum frequency
Deltaf=0.1; % frequency sampling resolution
f=0:Deltaf:fmax; % generate +ve frequencies starting from 0
f=[fliplr(-f(2:end)),f(1:end)]; % add -ve frequencies;
fmax=max(f); % change fmax if fmax is not an integer multiple of Deltaf
fposind=find(f>=0); % indexes to +ve frequencies
fnegind=find(f<0); % indexes to -ve frequencies

mev=0; % initialise max eigenvalue to zero

Lambda=zeros(Npts^2,Npts^2,length(f));
Gamma=zeros(Npts^2,Npts^2,length(f));
T=zeros(Npts^2,Npts^2,length(f));

perc=0;
fprintf('\nConstructing direct propagtor: ')
% Now construct Gamma and Lambda
for fi=1:length(f)
    fprintf(repmat('\b',1,length(perc))); % Display percentage complete
    perc=num2str(fi/length(f));
    fprintf(perc);
    for i=1:Npts^2
        for j=1:Npts^2
            L=(1+sqrt(-1)*2*pi*f(fi)/alpha)^-1*(1+sqrt(-1)*2*pi*f(fi)/beta)^-1;
            q=(1+sqrt(-1)*2*pi*f(fi))/gamma_e;
            
            rx=rad*min([wrapTo2Pi(anglesX(j)-anglesX(i)),wrapTo2Pi(anglesX(i)-anglesX(j))]);
            ry=rad*min([wrapTo2Pi(anglesY(j)-anglesY(i)),wrapTo2Pi(anglesY(i)-anglesY(j))]);
            r=sqrt(rx^2+ry^2);
            
            if i~=j % no self propagation
                Gammatemp=dX*r_e^2/(2*pi)*besselk(0,r*q/r_e);
            else
                Gammatemp=0;
            end
            
            Lambda(j,i,fi)=G(j,i)*Gammatemp*L;
            Gamma(j,i,fi)=G(j,i)*Gammatemp;
            
        end
    end
    mev(fi)=max(real(eig(Lambda(:,:,fi)))); % save maximum eigenvalue
end

Lambda=0.85*Lambda/max(real(mev)); % scale so max eigenvalue is 0.85 (must be less than one for stability)
Gamma=0.85*Gamma/max(real(mev)); % scale so max eigenvalue is 0.85 (must be less than one for stability)


%% now construct transfer function
T=[];
for i=1:length(f)
    T(:,:,i)=inv(eye(Npts^2)-Lambda(:,:,i));
end
%% Now construct the spectral density matrix (frequency domain correlation matrix)
S=[];
for i=1:length(f)
   S(:,:,i)=T(:,:,i)*(T(:,:,i)');
end
%% Now do spectral factorization to recover the transfer function
[T_est,Z] = specfactorization_wilson(S(:,:,fposind),fmax); % Input positive frequencies to the Wilson algorithm
T_est=cat(3,flip(conj(T_est(:,:,2:end)),3),T_est); % add negative freqency components to T
epsilon=sqrt(sum(sum(sum(abs(T-T_est).^2))))/sqrt(sum(sum(sum(abs(T).^2)))) % compute the error between the actual and estimated transfer functions
%% Now compute estimated Lambda and Gamma
Gamma_est=zeros(size(T));
Lambda_est=zeros(size(T));
for i=1:size(T,3)
    Lambda_est(:,:,i)=eye(Npts^2)-inv(T_est(:,:,i));
    L=(1+sqrt(-1)*2*pi*f(i)/alpha)^-1*(1+sqrt(-1)*2*pi*f(i)/beta)^-1;
    Gamma_est(:,:,i)=Lambda_est(:,:,i)/L;
end


%% Now inverse Fourier transform to obtain temporal quantities
Lambdatemp=2*fmax*ifft(ifftshift(Lambda,3),[],3,'symmetric');
Gammatemp=2*fmax*ifft(ifftshift(Gamma,3),[],3,'symmetric');
Ttemp=2*fmax*ifft(ifftshift(T,3),[],3,'symmetric');

Lambda_esttemp=2*fmax*ifft(ifftshift(Lambda_est,3),[],3,'symmetric');
Gamma_esttemp=2*fmax*ifft(ifftshift(Gamma_est,3),[],3,'symmetric');
T_esttemp=2*fmax*ifft(ifftshift(T_est,3),[],3,'symmetric');

%% Plot one elements of T in time
figno=1;
figure(figno)
clf
hold on
lw=1; % line width for plots
fsz=12; % font size

t=(0:(length(f)-1))/(2*fmax); % time points
tmax=max(t);
N=zeros(1,Npts^2)'; % input vector
N(1)=1; % input to first point only

Q=zeros(1,length(t));
Qalg=zeros(1,length(t));
point=Npts; % choose end point for plotting

% First plot T
for ti=1:length(t)
    Qtemp=Ttemp(:,:,ti)*N; % actual system activity
    Q(ti)=Qtemp(point);
    
    Qtemp=T_esttemp(:,:,ti)*N; % estimated system activity
    Qalg(ti)=Qtemp(point);
    
end
subplot(3,2,1)
plot(t,Q,'k','LineWidth',lw)
hold on

xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$T(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
title('(a)','interpreter','latex')
ylim([1.1*min([Q,Qalg]),1.1*max([Q,Qalg])])
xlim([0,tmax])

axes('Position',[.2 .76 .24 .14])
box on
ti=find(t<0.1);
plot(t(ti),Q(ti),'k','LineWidth',lw)
set(gca,'XLim',[t(min(ti)),t(max(ti))])
set(gca,'YLim',[1.1*min(Q(ti)),1.1*max(Q(ti))])
set(gca,'TickLength',[0.04, 0.01])

subplot(3,2,2)
plot(t,Q-Qalg,'k','LineWidth',lw)
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$T(\tau)-T_{\rm est}(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
ylim([1.1*min(Q-Qalg),1.1*max(Q-Qalg)])
xlim([0,tmax])
title('(b)','interpreter','latex')

% Now plot Lambda
for ti=1:length(t)
    Qtemp=Lambdatemp(:,:,ti)*N;
    Q(ti)=Qtemp(point);
    
    Qtemp=Lambda_esttemp(:,:,ti)*N;
    Qalg(ti)=Qtemp(point);
    
end
subplot(3,2,3)
plot(t,Q,'k','LineWidth',lw)
hold on

xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Lambda(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
title('(c)','interpreter','latex')
ylim([1.1*min([Q,Qalg]),1.1*max([Q,Qalg])])
xlim([0,tmax])

axes('Position',[.2 .46 .24 .14])
box on
ti=find(t<0.1);
plot(t(ti),Q(ti),'k','LineWidth',lw)
set(gca,'XLim',[t(min(ti)),t(max(ti))])
set(gca,'YLim',[1.1*min(Q(ti)),1.1*max(Q(ti))])
set(gca,'TickLength',[0.04, 0.01])

subplot(3,2,4)
plot(t,Q-Qalg,'k','LineWidth',lw)
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Lambda(\tau)-\Lambda_{\rm est}(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
ylim([1.1*min(Q-Qalg),1.1*max(Q-Qalg)])
xlim([0,tmax])
title('(d)','interpreter','latex')

% Now plot Gamma
for ti=1:length(t)
    Qtemp=Gammatemp(:,:,ti)*N;
    Q(ti)=Qtemp(point);
    
    Qtemp=Gamma_esttemp(:,:,ti)*N;
    Qalg(ti)=Qtemp(point);
    
end
subplot(3,2,5)
plot(t,Q,'k','LineWidth',lw)
hold on

xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Gamma(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
title('(e)','interpreter','latex')
ylim([1.1*min([Q,Qalg]),1.1*max([Q,Qalg])])
xlim([0,tmax])

axes('Position',[.2 .175 .24 .125])
box on
ti=find(t<0.1);
plot(t(ti),Q(ti),'k','LineWidth',lw)
set(gca,'XLim',[0,t(max(ti))])
set(gca,'YLim',[1.1*min(Q),1.1*max(Q(ti))])
set(gca,'TickLength',[0.04, 0.01])

subplot(3,2,6)
plot(t,Q-Qalg,'k','LineWidth',lw)
xlabel('$\tau$ (s)','interpreter','latex')
ylabel('$\Gamma(\tau)-\Gamma_{\rm est}(\tau)$','interpreter','latex')
box on
set(gca,'FontSize',fsz)
set(gca,'TickLength',[0.04, 0.01])
ylim([1.1*min(Q-Qalg),1.1*max(Q-Qalg)])
xlim([0,tmax])
title('(f)','interpreter','latex')

%% Now Plot Spatial spread of T, Lambda, Gamma
figure(2)
clf
hold on
lw=1; % plot line width
fontsz=16; % font size
fw=0.38/2; % plot frame width
fh=0.38/2; % plot frame height
cols=['r','b','g']; % some line colors
cmap=colormap('jet'); % choose colormap
alphlab={'(a)','(b)','(c)'}; % some frame labels

N=zeros(1,Npts^2)'; % input vector
N(floor(Npts/2)*(Npts)+ceil(Npts/2))=1; % input at the central point

[X,Y]=meshgrid(theta,theta); % create 2D grid of points

tind=[2,3,4]; % choose some time points to plot
tplot=t(tind);


% Do T plots first
% find max/min values to scale colorbars
maxQ=-inf;
minQ=inf;
for tpi=1:length(tplot)
    Q=Ttemp(:,:,tind(tpi))*N;
    if max(Q)>maxQ
        maxQ=max(Q);
    end
end

for tpi=1:length(tplot)
    Q=Ttemp(:,:,tind(tpi))*N;
    
    if min(Q)<minQ
        minQ=min(Q);
    end
end

% Now do the plots
for tpi=1:length(tplot)
    subplot(3,3,tpi)
    
    Q=Ttemp(:,:,tind(tpi))*N;
    Q=reshape(Q,[Npts,Npts]);
    
    imagesc(Q,[minQ,maxQ])
    
    set(gca,'Position', [0.05+(.05+fw)*(tpi-1),0.66,fw,fh])
    axis equal
    axis off
    title(['$\tau=~$',num2str(round(tplot(tpi)*1e4)/10),'$~\rm ms$'],'interpreter','latex')
    
    xlh=xlabel(alphlab{tpi},'interpreter','latex','FontSize',fontsz);
    xlh.Position(2)=xlh.Position(2)-4;
    set(get(gca,'XLabel'),'Visible','on')
    if tpi==1
        ylh=ylabel('$T(\tau)$','interpreter','latex','FontSize',fontsz);
        ylh.Position(1)=ylh.Position(1)+7;
        set(get(gca,'YLabel'),'Visible','on')
    end
    set(gca,'FontSize',fontsz)
    
end

colorbar('Position',[0.75,.66,.025,fh]);

% Now plot Lambda
alphlab={'(d)','(e)','(f)'};
maxQ=-inf;
minQ=inf;
for tpi=1:length(tplot)
    Q=Lambdatemp(:,:,tind(tpi))*N;
    if max(Q)>maxQ
        maxQ=max(Q);
    end
end

for tpi=1:length(tplot)
    Q=Lambdatemp(:,:,tind(tpi))*N;
    
    if min(Q)<minQ
        minQ=min(Q);
    end
end
for tpi=1:length(tplot)
    subplot(3,3,tpi+3)
    
    Q=Lambdatemp(:,:,tind(tpi))*N;
    Q=reshape(Q,[Npts,Npts]);
    imagesc(Q,[minQ,maxQ])
    
    set(gca,'Position', [0.05+(.05+fw)*(tpi-1),0.38,fw,fh])
    axis equal
    axis off
    xlh=xlabel(alphlab{tpi},'interpreter','latex','FontSize',fontsz);
    xlh.Position(2)=xlh.Position(2)-4;
    set(get(gca,'XLabel'),'Visible','on')
    if tpi==1
        ylh=ylabel('$\Lambda(\tau)$','interpreter','latex','FontSize',fontsz);
        ylh.Position(1)=ylh.Position(1)+7;
        set(get(gca,'YLabel'),'Visible','on')
    end
    set(gca,'FontSize',fontsz)
end

colorbar('Position',[0.75,.38,.025,fh]);

% Now plot Gamma
alphlab={'(g)','(h)','(i)'};
maxQ=-inf;
minQ=inf;
for tpi=1:length(tplot)
    Q=Gammatemp(:,:,tind(tpi))*N;
    if max(Q)>maxQ
        maxQ=max(Q);
    end
end

for tpi=1:length(tplot)
    Q=Gammatemp(:,:,tind(tpi))*N;
    if min(Q)<minQ
        minQ=min(Q);
    end
end
for tpi=1:length(tplot)
    subplot(3,3,tpi+6)
    
    
    Q=Gammatemp(:,:,tind(tpi))*N;
    Q=reshape(Q,[Npts,Npts]);
    
    imagesc(Q,[minQ,maxQ])
    
    set(gca,'Position', [0.05+(.05+fw)*(tpi-1),0.1,fw,fh])
    axis equal
    axis off
    set(gca,'FontSize',fontsz)
    xlh=xlabel(alphlab{tpi},'interpreter','latex','FontSize',fontsz);
    xlh.Position(2)=xlh.Position(2)-4;
    set(get(gca,'XLabel'),'Visible','on')
    if tpi==1
        ylh=ylabel('$\Gamma(\tau)$','interpreter','latex');
        ylh.Position(1)=ylh.Position(1)+9;
        set(get(gca,'YLabel'),'Visible','on')
    end
    
end
colorbar('Position',[0.75,.1,.025,fh]);

