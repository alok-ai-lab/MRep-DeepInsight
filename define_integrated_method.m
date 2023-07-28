function MethodIntegrated = define_integrated_method(Parm)

MethodIntegrated=[];

for j=1:length(Parm.integrate)
    if j==length(Parm.integrate)
        MethodIntegrated=[MethodIntegrated,Parm.integrate{j}];
    else
        MethodIntegrated=[MethodIntegrated,Parm.integrate{j},'+'];
    end
end