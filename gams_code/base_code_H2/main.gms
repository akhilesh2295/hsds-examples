option minlp = baron;
option optcr = 1e-4;
option optca = 1e-4;

Sets
t "Time stages" /1*%gams.user1%/
ren "Renewable energy sources" /wind, solar/
;

Scalars
delta_t "time resolution in hrs" /%gams.user2%/
time_hor "time horizon in consideration in hrs" /24/
m_dem "constant H2 demand (kg/hr)"/
$include pars/input_profiles/demh2.gms
/

ncpocell_h2 "Electrolyzer cell power input at max H2 production (kW)" /5.6/
mcell_max "Electrolyzer cell max H2 production rate (kg/hr)" /0.087/
ncpocomp_h2 "H2 compression specific power requirement (MWh/kg) 30 bar-->100 bar" /0.001/
coiv_tank "Unit capital cost of H2 storage ($/kg)" /516/
coof_tank "Unit FOM cost of H2 storage (fraction of capital cost)" /0.01/
coiv_el "Unit capital cost of electrolyzer ($/MW)" /800000/
coof_el "Unit FOM cost of electrolyzer (fraction of capital cost)" /0.07/ 
coiv_comp "Unit capital cost of compressor ($/MW)" /1200000/
coof_comp "Unit FOM cost of compressor (fraction of capital cost)" /0.04/
CRF_h2 "capital recovery factor of project" /0.103/
h2_pen "Penalty on unmet H2 demand ($/kg)" /1000000/
co2_pen "CO2 Penalty ($/ton)" /80/
grid_emint "CO2 emission intensity of grid electricity (ton/MWh)" /0.45/
;

Parameters
coiv(ren) "unit capital cost of renewable source ren ($/MW)"/   # need to update these cost parameters as well
wind 1000000
solar 1000000
solar_panel_eff 0.1 # need to find academic references for this and below parameters used
wind_coeff_of_perf 0.40 # https://thundersaidenergy.com/downloads/wind-power-impacts-of-larger-turbines/#:~:text=If%20you%20want%20a%20good,total%20available%20incoming%20wind%20energy.
rho_air = 1.293 # kg/m^3
blade_length = 52 # metres
/

grid_price(t) "grid electricity price ($/MWh)"/
$include pars/input_profiles/elec_price.gms
/

ghi(t) "ghi of solar at time t"/
$include pars/input_profiles/ghi.gms
/

wind_speed(t) "wind speed at time t"/
$include pars/input_profiles/wind_speed.gms
/

;

$include "pars/energy_stor/stor_pars.gms";

Positive variables
*system decisions
P_grid(t) "Decision:power purchased from grid at time t (MW)"
Pmax_sor(ren) "Decision:max installed capacity of renewable source ren (MW)"
P_sor(ren,t) "Decision: power generated by renewable source ren at time t (MW)"
ncp_stor "Decision: Nominal power capacity of energy storage (MW)"
m_el(t) "Decision:electrolyzer h2 production flowrate at time t (kg/hr)"
Pmax_el "Decision:installed electrolyzer capacity (MW)"
Pmax_comp "Decision:installed H2 compressor capacity (MW)"
mh2max_tank "Decision:max capacity of H2 storage (kg)"
m_out(t) "Decision:H2 output from system at time t (kg/hr)"
m_um(t) "Decision:Unmet H2 demand at time t (kg/hr)"

*Li-ion cell variables
yc(t) "Power charged to single li-ion cell at time t in W"
yd(t) "Power discharged from single li-ion cell at time t in W"
OCV_lib(t) "Open circuit voltage of li-ion cell in V"
Vcell_lib(t) "Li-ion terminal voltage in V"
soc_lib(t) "li-ion cell SOC"
ncell_lib "Total number of cells in battery"
E_cell(t) "Energy capacity of each cell (Wh)"

*Li-ion battery variables
nce_stor "nominal energy capacity of battery in MWh"
E_stor(t) "Energy capacity of storage technology i at time t (MWh)"
cov_stor(t) "Storage variable o&m cost ($)"
cov_stortot "Total Storage variable o&m cost ($)"
civ_stor "Storage investment cost scaled over scheduling horizon ($)"
cof_stor "Storage fixed o&m cost scaled over scheduling horizon ($)"
lcos "Levelized cost of storage for technology i ($/MWh)"

*H2 electrolyzer variables
Ncell_el "Number of electrolyzer cells"
m_norm(t) "Normalized H2 production rate of one electrolyzer cell at time t"
P_el(t) "Power consumed by electrolyzer at time t (MW)"
civ_el "total investment cost of h2 electrolyzer ($)"
cof_el "total fom cost of h2 electrolyzer ($)"

*Renewable farm variables
P_avl_sor(ren,t) "Renewable energy available at time t (MW)"
civ_sor(ren) "Investment cost of renewable source ren ($)"

*H2 compressor variables
P_comp(t) "Power consumed by compressor at time t (MW)"
civ_comp "Total capital cost of compressor ($)"
cof_comp "total fom cost of compressor ($)"

*H2 storage variables
m_tank(t) "Mass of H2 in storage tank at time t (kg)"
civ_tank "investment cost of h2 storage ($)"
cof_tank "fom cost of h2 storage ($)"

*Overall system variables
m_excess(t) "Excess H2 output from system at time t (kg/hr)"
cov_grid "Total cost of grid bought electricity ($)"
cov_um "Total penalty on unmet h2 demand ($)"
cov_co2 "emission penalty on grid bought electricity ($)"
civ_tot "total investment cost of system ($)"
cof_tot "total fom cost of system ($)"
cov_tot "total variable cost of system ($)"
lcoh "levelized cost of h2 ($/kg)"
adum

*Land allocation variables for solar and wind
land(ren) "land allocated for renewable (solar/wind) for the county"

;

Variable
flag_c(t) "Variable for charging operation of battery"
flag_d(t) "Variable for discharging operation of battery"
P_stor(t) "Decision: Power o/p of energy storage technology i at time t (MW): positive for discharge, negative for charge"
TC "Total system cost ($)"
dummy_obj
;

Binary variable
z_idle(t) "Binary variable for idle state of battery at time t"
;

Equation dummy_eq; dummy_eq.. dummy_obj =e= adum;
adum.up = 5;

*Energy storage - operational constraints
Equation flag_charge_eq; flag_charge_eq(t).. sqrt( power(( yd(t) - yc(t) ),2) )*flag_c(t) =e= ( sqrt( power(( yd(t) - yc(t) ),2) ) - yd(t) + yc(t)  ) / 2;
Equation flag_disc_eq; flag_disc_eq(t).. sqrt( power(( yd(t) - yc(t) ),2) )*flag_d(t) =e= ( sqrt( power(( yd(t) - yc(t) ),2) ) + yd(t) - yc(t)  ) / 2;
Equation combo_outeq; combo_outeq(t).. Vcell_lib(t) =e=  OCV_lib(t) - icell_lib*Rcell_lib*( -flag_c(t) + flag_d(t) )*(1 - z_idle(t) );
Equation ocveq_lib; ocveq_lib(t).. OCV_lib(t) =e= a1_lib*(soc_lib(t)**3) + a2_lib*(soc_lib(t)**2) + a3_lib*soc_lib(t) + a4_lib;
Equation new_powereq; new_powereq(t).. sqrt( power( ( yc(t) - yd(t) ), 2 ) ) =e= Vcell_lib(t)*icell_lib*(1 - z_idle(t) );
Equation pconn; pconn(t).. P_stor(t) =e= ncell_lib*power(10,-6)*( yd(t) - yc(t) );
Equation ncelleq_lib; ncelleq_lib.. ncell_lib =e= ncp_stor*(10**6)/ncpocell_lib;
Equation epratio_lib; epratio_lib.. nce_stor =e= ncho_lib*ncp_stor;
Equation lib_f2; lib_f2(t).. E_stor(t) =e= nce_stor*soc_lib(t);
Equation ebalcell; ebalcell(t)$(ord(t) ne card(t)).. E_stor(t+1) =e= E_stor(t) - ( eta_rte*flag_c(t) + flag_d(t) )*P_stor(t)*delta_T;
Equation ecellcy1; ecellcy1(t)$(ord(t) eq card(t)).. ( E_stor(t) - E_stor('1') )  =l= cyctol*E_stor('1'); 
Equation ecellcy2; ecellcy2(t)$(ord(t) eq card(t)).. ( E_stor(t) - E_stor('1') )  =g= -1*cyctol*E_stor('1'); 
Equation maxcapeq; maxcapeq(t).. E_stor(t) =l= nce_stor;
Equation pbndd1; pbndd1(t).. P_stor(t) =l= ncp_stor;
Equation pbndd2; pbndd2(t).. (-1)*P_stor(t) =l= ncp_stor;

*Energy storage - cost
Equation batinv; batinv.. civ_stor =e= coiv_stor*nce_stor*1000*crf_stor*time_hor/8760;
Equation batfix; batfix.. cof_stor =e= coof_stor*ncp_stor*1000*time_hor/8760;
Equation batvar1; batvar1(t).. cov_stor(t) =e= coov_stor*( yc(t) + yd(t) )*ncell_lib*power(10,-6)*delta_T;
Equation batvar3; batvar3.. cov_stortot =e= sum(t$(ord(t) ne card(t) ), cov_stor(t));

*Renewable farm constraints
* Equation pren_eq; pren_eq(ren,t)$(ord(t) ne card(t)).. P_sor(ren,t) =l= cf(ren,t)*Pmax_sor(ren);
* Equation pren_avl_eq; pren_avl_eq(ren,t)$(ord(t) ne card(t)).. P_avl_sor(ren,t) =e= cf(ren,t)*Pmax_sor(ren);
* Equation civ_reneq; civ_reneq(ren).. civ_sor(ren) =e= coiv(ren)*Pmax_sor(ren)*CRF_h2*time_hor/8760;

* Have split renewable constraints as solar and wind. If unable to debug and set ren == solar and ren == wind, then we can just declare them separate variables and move ahead. 

*Renewable farm constraints - solar only - akhilesh edited
Equation pren_eq_s; pren_eq_s(ren,t)$(ord(t) ne card(t)).. P_sor(ren,t) =l= solar_panel_eff*ghi(t)*land(ren)/1000;
Equation pren_avl_eq_s; pren_avl_eq_s(ren,t)$(ord(t) ne card(t)).. P_avl_sor(ren,t) =e= solar_panel_eff*ghi(t)*land(ren)/1000;  # eqn should be valid only for ren == solar
Equation civ_reneq_s; civ_reneq_s(ren).. civ_sor(ren) =e= coiv(ren)*Pmax_sor(ren)*CRF_h2*time_hor/8760; # eqn to be valid only for ren == solar

*Renewable farm constraints - wind only - akhilesh edited
Equation pren_eq_w; pren_eq_w(ren,t)$(ord(t) ne card(t)).. P_sor(ren,t) =l= 0.5*wind_coeff_of_perf*rho_air*pi*blade_length*blade_length*wind_speed(t)*wind_speed(t)*wind_speed(t)/1000; # P = 0.5 Cp ρ π R^2 V^3
Equation pren_avl_eq_w; pren_avl_eq_w(ren,t)$(ord(t) ne card(t)).. P_avl_sor(ren,t) =e= 0.5*wind_coeff_of_perf*rho_air*pi*blade_length*blade_length*wind_speed(t)*wind_speed(t)*wind_speed(t)/1000;  # eqn should be valid only for ren == wind
Equation land_wind; land_wind(ren).. land(ren) =e= P_avl_sor(ren,t)/2.5*0.125;  # eqn should be valid only for ren == wind
# 80 acres = 0.125 sq. mi.
# https://landgate.com/news/2023/04/07/does-my-land-qualify-for-a-wind-lease/#:~:text=Acreage%20Required%20for%20a%20Wind%20Farm&text=One%20wind%20turbine%20can%20require,on%20much%20of%20the%20land.
# note that land required for wind can still be used for agricultural purposes
# wind has quite a few more restrictions as can be seen on the reference link above
Equation civ_reneq_w; civ_reneq_w(ren).. civ_sor(ren) =e= coiv(ren)*Pmax_sor(ren)*CRF_h2*time_hor/8760;      # eqn should be valid only for ren == wind



*Electrolyzer constraints
Equation ncellel_eq; ncellel_eq.. Ncell_el =e= 1000*Pmax_el/ncpocell_h2;
Equation normh2_eq; normh2_eq(t).. m_norm(t)*Ncell_el =e= m_el(t)/mcell_max;
Equation pel_eq; pel_eq(t).. P_el(t) =e= ((-8.5231*m_norm(t)*m_norm(t)) + 23.995*m_norm(t) + 47.752)*m_el(t)*0.001;
Equation civel_eq; civel_eq.. civ_el =e= coiv_el*Pmax_el*CRF_h2*time_hor/8760;
Equation cofel_eq; cofel_eq.. cof_el =e= coof_el*civ_el;

*Compressor constraints
Equation pcomp_eq; pcomp_eq(t).. P_comp(t) =e= m_el(t)*ncpocomp_h2;
Equation pcompbnds; pcompbnds(t).. P_comp(t) =l= Pmax_comp;
Equation civcomp_eq; civcomp_eq.. civ_comp =e= coiv_comp*Pmax_comp*CRF_h2*time_hor/8760;
Equation cofcomp_eq; cofcomp_eq.. cof_comp =e= coof_comp*civ_comp;

*H2 storage constraints
Equation h2tankbal; h2tankbal(t)$(ord(t) ne card(t)).. m_tank(t+1) =e= m_tank(t) + (m_el(t) - m_out(t))*delta_T;
Equation h2tankcyc; h2tankcyc(t)$(ord(t) eq card(t)).. m_tank(t) =e= m_tank('1');
Equation h2tankub; h2tankub(t).. m_tank(t) =l= mh2max_tank;
Equation civtank_eq; civtank_eq.. civ_tank =e= coiv_tank*mh2max_tank*CRF_h2*time_hor/8760;
Equation coftank_eq; coftank_eq.. cof_tank =e= coof_tank*civ_tank;

*Overall system equations
Equation energybal; energybal(t)$(ord(t) ne card(t)).. P_grid(t) + sum(ren, P_sor(ren,t)) + P_stor(t) =e= P_el(t) + P_comp(t);
Equation h2bal; h2bal(t)$(ord(t) ne card(t)).. m_dem =e= m_out(t) + m_um(t) - m_excess(t);
Equation covgrid_eq; covgrid_eq.. cov_grid =e= sum(t$(ord(t) ne card(t)), grid_price(t)*P_grid(t)*delta_T);
Equation covum_eq; covum_eq.. cov_um =e= sum(t$(ord(t) ne card(t)), h2_pen*m_um(t)*delta_T);
Equation covco2_eq; covco2_eq.. cov_co2 =e= sum(t$(ord(t) ne card(t)), P_grid(t)*co2_pen*grid_emint*delta_T);
Equation civtot_eq; civtot_eq.. civ_tot =e= sum(ren, civ_sor(ren)) + civ_el + civ_comp + civ_tank + civ_stor;
Equation coftot_eq; coftot_eq.. cof_tot =e= cof_el + cof_comp + cof_tank + cof_stor;
Equation covtot_eq; covtot_eq.. cov_tot =e= cov_grid + cov_um + cov_co2 + cov_stortot;
Equation lcoh_eq; lcoh_eq.. lcoh*( sum(t$(ord(t) ne card(t)), m_out(t)) ) =e= civ_tot + cof_tot + cov_tot - cov_um;
Equation totcost_eq; totcost_eq.. TC =e= civ_tot + cof_tot + cov_tot;

$include "bounds.gms";

model storage /all/;
option reslim = 30;

*solve storage using MINLP maximizing dummy_obj;
solve storage using MINLP minimizing TC;

*****Post processing******
execute_unload "results.gdx";
execute "gdx2sqlite -i results.gdx -o results.db";

