# raw-data-for-comb-testing
Supplementary raw data for paper titled *COMB: Scalable Concession-driven Opponent Models using Bayesian Learning for Preference Learning in Bilateral Multi-issue Automated Negotiation* 

## Level 1 Folders: Methods

1. **SOTA**: <i>CUHKagent Value Model</i> and <i>Hardheaded Frequency Model</i>. In files of this folder, the first and second colomns are datas of CUHK and Hard, respectively.  
5. **Existing_Bayesian/Specific**: the exsiting bayesian model applying the <i>Specific Bidding Utility</i> likelihood function.  
5. **Existing_Bayesian/Stepwise**: the exsiting bayesian model applying the <i>Stepwise Concession</i> likelihood function.  
2. **COMB/Specific**: the proposed COMB model applying the <i>Specific Bidding Utility</i> likelihood function.
2. **COMB/Stepwise**: the proposed COMB model applying the <i>Stepwise Concession</i> likelihood function.
3. **COMB/Regression**: the proposed COMB model applying the <i>Linear Regression</i> likelihood function.  
4. **COMB/Expectation**: the proposed COMB model applying the <i>Expectation Concession</i> likelihood function.   


## Level 2 Folders: Opponent category

1. **ANAC**: ANAC agents.  
2. **noise_0**: basic agents with 0 noise.  
3. **noise_0.005**: basic agents wth 0.005 noise.  

## Level 3 Folders: Opponent Name

1. **Time0.1-10**: Time-dependent agents with parameter equals to 0.1, 1, 2, 5, and 10, respectively.  
2. **Offset0.7-0.9**: Offset agents with parameter equals to 0.7, 0.8, and 0.9, respectively.  
3. **Reserve0.3-0.9**: Reservation agents with parameter equals to 0.3, 0.5, 0.7 and 0.9, respectively.
4. **AgentK - Yushu**: Corresponding ANAC agents.


## Level 4 files: Accuricies
Named by: {domain_name}_{Opposition Feature}{Distribution Feature}.json



