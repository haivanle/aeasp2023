clear

cd "/Users/haivanle/Documents/AEA/"

* Compensation - Total SEC - With Policy Dummies and Volatility Index *
// ssc install estout, replace

import delimited "zscore2024v2.csv"

** First Stage **
reg adjusted_tc emp tenure payczar doddfrank cbvix mom, vce(robust) // First Stage IV 

test emp tenure // test instrument

est store reg1

predict ajtc_hat, xb // resid 

reg ratio_stock_options emp tenure payczar doddfrank cbvix mom, vce(robust) // First Stage IV 

test emp tenure

est store reg2

predict rso_hat, xb // resid 

** Second Stage **
reg zscore ajtc_hat rso_hat payczar doddfrank cbvix mom, vce(robust) // Second Stage IV

test payczar doddfrank

test cbvix mom

test payczar doddfrank cbvix mom

est store reg_2SLS

esttab reg1 reg2 reg_2SLS, b(3) se(3) star compress nogap s(N r2)

