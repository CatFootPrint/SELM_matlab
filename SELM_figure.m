x=1:1:10;
L=((5*x+2)+sqrt((5*x-8/5).^2+36/25))/6;
plot(x,L,'-o');
grid on;
xlabel('$M$','Interpreter','LaTex');ylabel('$\frac{5M+2+\sqrt{\left(5M-\frac{8}{5}\right)^2+\frac{36}{25}}}{6}$','Interpreter','LaTex');