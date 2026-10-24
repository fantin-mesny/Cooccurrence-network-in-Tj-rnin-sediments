import pandas as pd
import networkx as nx
from cdlib import algorithms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fa2 import ForceAtlas2
from scipy import stats
from statsmodels.stats import multitest
import numpy as np
import itertools


def flatten(xss):
    return [x for xs in xss for x in xs]

def sample_non_overlapping(comms,random_state):
    """
    Sample non-overlapping sets from lst.
    Assumes len(lst) >= sum(lstOfSizes)
    """
    lst=list(itertools.chain.from_iterable(comms))
    lstOfSizes=[len(com) for com in comms]
    lst_subset=pd.DataFrame(lst)
    output=[]
    for i in range(len(lstOfSizes)):
        sample=list(lst_subset[0].sample(lstOfSizes[i],random_state=random_state))
        lst_subset=lst_subset[~(lst_subset[0].isin(sample))]
        output.append(sample)
    return output

def parseSparccOutputs(dir): #parse SparCC outputs and build network
    coocc=pd.read_csv(dir+'/median_correlation.tsv',sep='\t').set_index('#OTU ID').stack().reset_index(drop=False).rename(columns={'#OTU ID':'node 1','level_1':'node 2',0:'correlation'})
    pvals=pd.read_csv(dir+'/pvalues.tsv',sep='\t').set_index('#OTU ID').stack().reset_index(drop=False).rename(columns={'#OTU ID':'node 1','level_1':'node 2',0:'p-value'})
    coocc=coocc.merge(pvals,on=['node 1','node 2'])

    coocc_pos_signif=coocc[(coocc['p-value']<0.01) & (coocc['correlation']>0.5)] # only significant positive co-occurrence associations
    print('Numb of positive co-occurrence associations:',len(coocc_pos_signif))
    network=nx.from_pandas_edgelist(coocc_pos_signif, source='node 1', target='node 2', edge_attr='correlation') # positive co-occurrence network

    return network,coocc_pos_signif,coocc

def getCommunities(G,reso=1.0): #identify network modules based on Louvain partition
    Louvain_partition = algorithms.louvain(G,weight='correlation',resolution=reso)
    return Louvain_partition.communities

def sortCommunitiesByEmergenceDate(comms): #check the year of the maximum abundance of communities to sort them
    defineOrderComms=pd.DataFrame()
    N=0
    for com in comms:
        otu_table_com=pd.DataFrame(otu_table[otu_table.index.isin(com)].mean(axis=0)).reset_index(drop=False)
        defineOrderComms.loc[comms.index(com),'peak']=otu_table_com[0].idxmax()
        #print(N,otu_table_com.loc[otu_table_com[0].idxmax(),'index'])
        N+=1
    defineOrderComms=defineOrderComms.sort_values(by='peak',ascending=False)
    #print(defineOrderComms)
    comms_filtered_ordered=[]
    for i in defineOrderComms.index:
        comms_filtered_ordered.append(comms[i])
    return comms_filtered_ordered

def saveComms(comms,outputFile): #save modules to file (for reproducibility)
    with open(outputFile,'w+') as outp:
        for comm in comms:
            outp.write(';'.join(comm)+'\n')

def parseComms(inputFile): #load previously identified modules
    listOfComms=[]
    with open(inputFile,'r') as inp:
        lines=inp.readlines()
        for line in lines:
            listOfComms.append(line[:-1].split(';'))
    return listOfComms

def findModuleHubs(G,comms):
    figHubs,ax=plt.subplots(2,4,figsize=(10,5),sharex=True,sharey=True)
    moduleKeystones_df=[]
    for com in comms:
        N=comms.index(com)
        name='Module '+str(comms.index(com))
        module = nx.subgraph(G, com)
        nodeProps_com=pd.concat([pd.DataFrame(nx.betweenness_centrality(module),index=['Betweenness']).T,pd.DataFrame(nx.closeness_centrality(module),index=['Closeness']).T,pd.DataFrame(nx.degree_centrality(module),index=['Degree']).T],axis=1)
        if N<=3:
            axToPlot_1=0
        else:
            axToPlot_1=1
        if N>3:
            axToPlot_2=N-4
        else:
            axToPlot_2=N-0
        sns.scatterplot(x='Degree',y='Closeness',data=nodeProps_com,palette='Greys',ax=ax[axToPlot_1][axToPlot_2],color='black',s=10)
        ax[axToPlot_1][axToPlot_2].set_title(name)

        closeness_threshold=nodeProps_com['Closeness'].quantile(0.95)
        degree_threshold=nodeProps_com['Degree'].quantile(0.95)
        moduleKeystones=nodeProps_com[(nodeProps_com['Degree']>degree_threshold) & (nodeProps_com['Closeness']>closeness_threshold)]
        moduleKeystones['Degree x Closeness']=moduleKeystones['Degree']*moduleKeystones['Closeness']
        moduleKeystones['Module']=name

        ax[axToPlot_1][axToPlot_2].axvline(degree_threshold,color='black')
        ax[axToPlot_1][axToPlot_2].axhline(closeness_threshold,color='black')
        ax[axToPlot_1][axToPlot_2].text(0.9,closeness_threshold,str(len(moduleKeystones)),ha='right',va='bottom')
        moduleKeystones_df.append(moduleKeystones)
    moduleKeystones_df=pd.concat(moduleKeystones_df)
#    moduleKeystones_df.to_csv('./module_hubs.csv')
#    figHubs.savefig(group+'/hubs.pdf')
    plt.close()
    return moduleKeystones_df


def testForTaxonomicOverrepresentation(setOfNodes,taxonomy,phyloRanks=['domain','phylum','class','order','family']):
    outputs=[]
    for rank in phyloRanks:
        inSet=taxonomy[taxonomy.index.isin(setOfNodes)]
        notInSet=taxonomy[~(taxonomy.index.isin(setOfNodes))]
        inSet_vc=inSet[rank].value_counts().to_dict()
        notInSet_vc=notInSet[rank].value_counts().to_dict()
        tests=pd.DataFrame(index=list(set(taxonomy[rank])),columns=['rank','stat','p-value','ratio in set','ratio in others'])
        for taxon in tests.index:
            if taxon not in inSet_vc:
                inSet_vc[taxon]=0
            if taxon not in notInSet_vc:
                notInSet_vc[taxon]=0        
            tests.loc[taxon,'rank']=rank
            tests.loc[taxon,'stat'],tests.loc[taxon,'p-value']=stats.fisher_exact([[inSet_vc[taxon],notInSet_vc[taxon]],[len(inSet)-inSet_vc[taxon], len(notInSet)-notInSet_vc[taxon]]],alternative='greater')
            tests.loc[taxon,'set']=';'.join(list(inSet[inSet[rank]==taxon].index))
            tests.loc[taxon,'ratio in set']=str(inSet_vc[taxon])+'/'+str(len(inSet))
            tests.loc[taxon,'ratio in others']=str(notInSet_vc[taxon])+'/'+str(len(notInSet))
        tests=tests.sort_values(by='p-value',ascending=True)
        tests['FDR']=multitest.multipletests(tests['p-value'],method='fdr_bh')[1]
        tests['-logFDR']=-tests['FDR'].astype(float).apply(np.log)
        outputs.append(tests)
    outputs=pd.concat(outputs)
    return outputs

def removeRedundantHigherRanks(overrepresentationResults):
    ranks=['domain','phylum','class','order','family'][::-1]
    for rank in ranks:
        rankData=overrepresentationResults[overrepresentationResults['rank']==rank]
        for ind in rankData.index:
            nodeSet=set(rankData.loc[ind,'set'].split(';'))
            for ind2 in overrepresentationResults[overrepresentationResults['rank'].isin(ranks[ranks.index(rank)+1:])].index:
                otherNodeSet=set(overrepresentationResults.loc[ind2,'set'].split(';'))
                if len(otherNodeSet.intersection(nodeSet))==len(otherNodeSet):
                    overrepresentationResults=overrepresentationResults.drop(index=ind2)
    return overrepresentationResults

def taxonomicOverrepresentationInModules(comms,taxonomy,cmaps):
    taxoOverrep_dfs=[]
    N=0
    for com in comms:
        taxoOverrep=testForTaxonomicOverrepresentation(com,taxonomy)
        taxoOverrep=taxoOverrep[taxoOverrep['FDR']<0.05]
        taxoOverrep_dfs.append(removeRedundantHigherRanks(taxoOverrep).reset_index(drop=False))
        taxoOverrep_dfs[-1]['Module']=str(N)
        taxoOverrep_dfs[-1]['Color']=cmaps[len(comms)][N]
        N+=1
    taxoOverrep_dfs=pd.concat(taxoOverrep_dfs).reset_index(drop=True)
    taxoOverrep_dfs['index']=taxoOverrep_dfs['Module']+' ('+taxoOverrep_dfs['rank'].str[0]+') '+taxoOverrep_dfs['index']
    return taxoOverrep_dfs

def getNetworkTopologicalFeatures(G,name):
    return {
        'Network':name,
        'Node number':G.number_of_nodes(),
        'Edge number': G.number_of_edges(),
        'Average degree':pd.DataFrame(G.degree())[1].mean(),
        'Network density':nx.density(G),
        'Clustering coefficient':nx.average_clustering(G),
    }

def getTopologicalPropertiesOfModules(G,comms):
    topologicalFeatures=[]
    name=0
    for com in comms:
        module = nx.subgraph(G, com)
        topologicalFeatures.append(getNetworkTopologicalFeatures(module,name))
        name+=1
    topologicalFeatures=pd.DataFrame(topologicalFeatures)
    return topologicalFeatures

def classifyProkaryotesWithGTDB(prokaryote_taxonomy):
    gtdb=pd.read_csv('bac120_metadata_r226.tsv',sep='\t')
    gtdb=gtdb[['gtdb_taxonomy']+[c for c in gtdb.columns if 'isol' in c or 'source' in c]]
    sources={'soil':['soil','litter'],'freshwater':['spring','freshwater','river','pond','lake'], 'marine water':['salt','ocean','sea','marine'],'plant':['root','rhizosphere','leaf','leaves','stem','plant','nodule','maize','wheat','rhizo','phyllo','endophyt'],'animal':['rumen','cattle','gut','skin','human'],'sediment':['sediment'],'pollution':['sewage','waste','polluted','landfill','trash']}
    outputDf=[]
    for genus in prokaryote_taxonomy.index:
        outputDf.append({s:0 for s in sources})
        outputDf[-1]['genus']=genus
        for source in set(gtdb[gtdb['gtdb_taxonomy'].str.contains('g__'+genus)]['ncbi_isolation_source']):
            for key in sources:
                for keyword in sources[key]:
                    if keyword in str(source).lower():
                        outputDf[-1][key]=1
    df=pd.DataFrame(outputDf).set_index('genus')
    df.to_csv('gtdb_taxa_classif.csv')
    return df

def testForEcologyOverrepresentation(setOfNodes,ecologyTable,ecoGroups):
    outputs=[]
    tests=pd.DataFrame(index=ecoGroups,columns=['stat','p-value','ratio in set','ratio in others'])
    for niche in ecoGroups:
        inSet=ecologyTable[ecologyTable.index.isin(setOfNodes)]
        notInSet=ecologyTable[~(ecologyTable.index.isin(setOfNodes))]
        inSet_vc=inSet[niche].value_counts().to_dict()
        notInSet_vc=notInSet[niche].value_counts().to_dict()
        for val in [0,1]:
            if val not in inSet_vc:
                inSet_vc[val]=0
            if val not in notInSet_vc:
                notInSet_vc[val]=0
        tests.loc[niche,'stat'],tests.loc[niche,'p-value']=stats.fisher_exact([[inSet_vc[1],notInSet_vc[1]],[inSet_vc[0],notInSet_vc[0]]],alternative='greater')
        tests.loc[niche,'set']=';'.join(list(inSet[inSet[niche]==1].index))
        tests.loc[niche,'ratio in set']=str(inSet_vc[1])+'/'+str(len(inSet))
        tests.loc[niche,'ratio in others']=str(notInSet_vc[1])+'/'+str(len(notInSet))
    tests=tests.sort_values(by='p-value',ascending=True)
    tests['FDR']=multitest.multipletests(tests['p-value'],method='fdr_bh')[1]
    tests['-logFDR']=-tests['FDR'].astype(float).apply(np.log)
    return tests

def testCommunitiesForEcologyOverrepresentation(comms,gtdb,ecoGroups):
    hm_df=[]
    tests=[]
    for comm in comms:
        commN=comms.index(comm)
        test=testForEcologyOverrepresentation(comm,gtdb,ecoGroups)
        test['module']=str(commN)
        tests.append(test)
        hm_df.append(test[['-logFDR']].rename(columns={'-logFDR':str(commN)}))
    ecologicalOverrepHM=pd.concat(hm_df,axis=1).T
    ecologicalOverrepTest=pd.concat(tests)
    ecologicalOverrepTest=ecologicalOverrepTest.reset_index()
    ecologicalOverrepTest=ecologicalOverrepTest.set_index(['index','module'])[['FDR']]
    ecologicalOverrepSignif=ecologicalOverrepHM.copy()
    for ind in ecologicalOverrepSignif.index:
        for col in ecologicalOverrepSignif.columns:
            if ecologicalOverrepTest.loc[(col,ind),'FDR']<0.05:
                ecologicalOverrepSignif.loc[ind,col]='*'
            else:
                ecologicalOverrepSignif.loc[ind,col]=''
    return ecologicalOverrepHM,ecologicalOverrepSignif

def connectedCommunities(corr,comms,interCommunityOnly):
    node_to_module = {}
    for i, group in enumerate(comms):
        for node in group:
            node_to_module[node] = i
    corr['module_node 1']=corr['node 1'].map(node_to_module).fillna(99)
    corr['module_node 1']=corr['module_node 1'].astype(int)
    corr['module_node 2']=corr['node 2'].map(node_to_module).fillna(99)
    corr['module_node 2']=corr['module_node 2'].astype(int)
    
    if interCommunityOnly:
        for ind in corr.index:
            if corr.loc[ind,'module_node 1']!=corr.loc[ind,'module_node 2']:
                corr.loc[ind,'connected communities']=str(min([corr.loc[ind,'module_node 1'],corr.loc[ind,'module_node 2']]))+'_'+str(max([corr.loc[ind,'module_node 1'],corr.loc[ind,'module_node 2']]))
    else:
        for ind in corr.index:
                corr.loc[ind,'connected communities']=str(min([corr.loc[ind,'module_node 1'],corr.loc[ind,'module_node 2']]))+'_'+str(max([corr.loc[ind,'module_node 1'],corr.loc[ind,'module_node 2']]))
    corr['inter-module']=corr['module_node 1']!=corr['module_node 2']
    return corr

def getEdgesCounts(corr_pos,corr_neg,comms):
    corr_neg=connectedCommunities(corr_neg,comms,True)
    corr_pos=connectedCommunities(corr_pos,comms,True)
    count_pos=corr_pos['connected communities'].value_counts()
    count_neg=corr_neg['connected communities'].value_counts()
    count_neg=pd.DataFrame(count_neg).rename(columns={'count':'negative edges'})
    count_pos=pd.DataFrame(count_pos).rename(columns={'count':'positive edges'})
    counts=count_pos.merge(count_neg,left_index=True,right_index=True,how='outer')
    counts=counts[~(counts.index.str.contains('_100'))].fillna(0)
    return counts


def nullModelOnEdgeNumbers(coocc, counts, corr_pos, corr_neg,sizeNull=10000):
    coocc_4null=coocc[coocc['node 1']!=coocc['node 2']].reset_index(drop=True)
    outputs=[]
    for i in range(sizeNull):
        print('   null sampling=',i)
        corr_null=pd.concat([coocc_4null[['node 1','node 2']],coocc_4null[['correlation','p-value']].sample(frac=1,random_state=i).reset_index(drop=True)],axis=1)
        corr_null_pos=corr_null[(corr_null['p-value']<0.01) & (corr_null['correlation']>0.5)]
        G_null=nx.from_pandas_edgelist(corr_null_pos, source='node 1', target='node 2', edge_attr='correlation')
        comms_null=getCommunities(G_null,1)
        comms_null=[comm for comm in comms_null if len(comm)>=10]
        corr_null_neg=corr_null[(corr_null['p-value']<0.01) & (corr_null['correlation']<-0.5)]
        counts_null=getEdgesCounts(corr_null_pos,corr_null_neg,comms_null)
        outputs.append({'null_index':i, 
                    'negative edges':sum(counts_null['negative edges'])/len(corr_null_neg),
                    'positive edges':sum(counts_null['positive edges'])/len(corr_null_pos)
                    })

    outputs.append({'null_index':'REAL','negative edges':sum(counts['negative edges'])/len(corr_neg),'positive edges':sum(counts['positive edges'])/len(corr_pos)})
    output_null1=pd.DataFrame(pd.DataFrame(outputs).set_index('null_index').stack()).reset_index(drop=False).rename(columns={'level_1':'Group',0:'Proportion of inter-module edges'})
    return output_null1

def nullModelOnModuleCompositions(counts,corr_pos,corr_neg,comms,sizeNull=10000):
    pvals=[]
    z_values=[]
    fakeComms_sets=[]
    pairsOfComms=list(set(counts['connected communities']))
    null_neg=[]
    null_pos=[]
    for i in range(sizeNull):
        print('   null sampling=',i)
        fakeComms=sample_non_overlapping(comms,random_state=i)
        fakeComms_sets.append(fakeComms)
        null_neg.append({pair:len(pd.concat([corr_neg[(corr_neg['node 1'].isin(fakeComms[int(pair.split('_')[0])])) & (corr_neg['node 2'].isin(fakeComms[int(pair.split('_')[1])]))],corr_neg[(corr_neg['node 2'].isin(fakeComms[int(pair.split('_')[0])])) & (corr_neg['node 1'].isin(fakeComms[int(pair.split('_')[1])]))]])) for pair in pairsOfComms if '99' not in pair})
        null_neg[-1]['index']=i
        null_pos.append({pair:len(pd.concat([corr_pos[(corr_pos['node 1'].isin(fakeComms[int(pair.split('_')[0])])) & (corr_pos['node 2'].isin(fakeComms[int(pair.split('_')[1])]))],corr_pos[(corr_pos['node 2'].isin(fakeComms[int(pair.split('_')[0])])) & (corr_pos['node 1'].isin(fakeComms[int(pair.split('_')[1])]))]])) for pair in pairsOfComms if '99' not in pair})
        null_pos[-1]['index']=i
    null_neg=pd.DataFrame(null_neg).set_index('index').T.merge(counts.set_index('connected communities')[['negative edges']],left_index=True,right_index=True)
    null_pos=pd.DataFrame(null_pos).set_index('index').T.merge(counts.set_index('connected communities')[['positive edges']],left_index=True,right_index=True)
    for ind in null_neg.index:
        null_vals=null_neg.loc[ind,list(range(sizeNull))]
        observed=null_neg.loc[ind,'negative edges']
        null_neg.loc[ind,'p-val']=np.mean(null_vals >= observed)
        if observed >= np.median(null_vals):
            # Observed in upper tail: use upper-tail probability
            p_upper = np.mean(null_vals >= observed)
            p_two_sided = 2 * p_upper
        else:
            # Observed in lower tail: use lower-tail probability
            p_lower = np.mean(null_vals <= observed)
            p_two_sided = 2 * p_lower
        # Cap at 1
        p_two_sided = min(p_two_sided, 1.0)
        null_neg.loc[ind,'p-val_2sided']=p_two_sided
    return null_pos,null_neg



cmaps={6:['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02'],7:['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'],5:['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']}
changePoints=[445,883,1260,1890,1960]

### Parse data and metadata, and build positive co-occurrence network
G,coocc_pos_signif,coocc=parseSparccOutputs('./data')
taxonomy=pd.read_csv('./data/taxonomy.tsv',sep='\t').set_index('genus')
otu_table=pd.read_csv('./data/otu_table.tsv',sep='\t').set_index('#OTU ID')

### Define communities
if os.path.isfile('./data/modules.txt'): # if modules were already computed (for full reproducibility)
    comms_filtered_ordered=parseComms('./data/modules.txt') # just load previously identified modules
    sortCommunitiesByEmergenceDate(comms_filtered_ordered)
else: #modules not already computed
    comms=getCommunities(G) # identify modules based on Louvain partition
    comms_filtered=[comm for comm in comms if len(comm)>=10] # sort communities by emergence/abundance over time (in "both" only, modules in prokaryote-only and eukaryote-only are numbered based on their overlap with "both")
    comms_filtered_ordered=sortCommunitiesByEmergenceDate(comms_filtered)
    saveComms('./modules_recomputed.txt') #save them to file (for reproducibility)

# remove nodes that were not placed in communities
notInComms=[nod for nod in G.nodes if nod not in flatten(comms_filtered_ordered)]
G.remove_nodes_from(notInComms)
coocc_pos_signif_filtered=coocc_pos_signif[~(coocc_pos_signif['node 1'].isin(notInComms))]
coocc_pos_signif_filtered=coocc_pos_signif_filtered[~(coocc_pos_signif_filtered['node 2'].isin(notInComms))]

### Identify hubs in modules
hubs=findModuleHubs(G,comms_filtered_ordered)

### Setup a figure
fig = plt.figure(figsize=(25, len(comms_filtered_ordered)+6))
gs = fig.add_gridspec(len(comms_filtered_ordered),7,width_ratios=[0.4,0.4,0.4,0.4,0.4, 1.6, 1],wspace=1.5)
ax0 = fig.add_subplot(gs[:-1, :-2])
ax0_props=[fig.add_subplot(gs[-1,i]) for i in range(5)]
axes_col1=[fig.add_subplot(gs[comms_filtered_ordered.index(com),-2]) for com in comms_filtered_ordered]
ax2 = fig.add_subplot(gs[:, -1])
    
### Plot network
fa=ForceAtlas2()
layout=fa.forceatlas2_networkx_layout(G, iterations=100)
nx.draw_networkx_edges(G,layout,ax=ax0, width=0.01,edgelist=list(zip(coocc_pos_signif_filtered['node 1'], coocc_pos_signif_filtered['node 2'])), edge_color='black')#, edge_color=corr['correlation'], edge_cmap=plt.cm.Greys)
for com in comms_filtered_ordered:
    nx.draw_networkx_nodes(G, layout, nodelist=com, node_color=[cmaps[len(comms_filtered_ordered)][comms_filtered_ordered.index(com)] for nod in com], alpha=1,ax=ax0,node_size=30, linewidths=0)
ax0.set_axis_off()

### Plot community abundance over time
otu_table_plot=otu_table.stack().reset_index(drop=False).rename(columns={'level_1':'Year',0:'Nreads'})
otu_table_plot['Year']=otu_table_plot['Year'].astype(float)
otu_table_plot=otu_table_plot.sort_values(by='Year')
N=0
for com in comms_filtered_ordered:
    sns.lineplot(data=otu_table_plot[otu_table_plot['#OTU ID'].isin(com)],x='Year',y='Nreads',ax=axes_col1[N],color=cmaps[len(comms_filtered_ordered)][comms_filtered_ordered.index(com)])
    N+=1
for ax in axes_col1[1:]:
    ax.sharex(axes_col1[0])
    
    for ax in axes_col1:
        for changePoint in changePoints:
            ax.axvline(changePoint,color='black')

axes_col1[0].set_title('Module abundance over time')
ax0_xlim=ax0.get_xlim()
ax0_ylim=ax0.get_ylim()
ax0.text(ax0_xlim[0],ax0_ylim[1],'%s nodes\n%s edges\n%s modules' % (str(len(G.nodes)),str(len(G.edges)),str(len(comms_filtered_ordered))),ha='left',va='top')
ax0.set_title('Positive co-occurrence network')

### Test modules for taxonomic enrichment
taxOverrep=taxonomicOverrepresentationInModules(comms_filtered_ordered,taxonomy,cmaps)
#plot and annotate a barplot:
sns.barplot(y='index',x='-logFDR',data=taxOverrep,palette=list(taxOverrep['Color']),ax=ax2)
for ind in taxOverrep.index:
    ax2.text(taxOverrep.loc[ind,'-logFDR']+1,ind,taxOverrep.loc[ind,'ratio in set'],ha='left',va='center')
    ax2.set_xlim(0,max(taxOverrep['-logFDR'])+2)
ax2.set_ylabel('')
ax2.set_title('Taxonomic enrichment\nin modules')
labels = [item.get_text() for item in ax2.get_yticklabels()]
ax2.set_yticklabels([' '.join(item.split(' ')[1:]) for item in labels])

### Calculate module topological properties 
moduleProps=getTopologicalPropertiesOfModules(G,comms_filtered_ordered)
N=0
for col in list(moduleProps.columns[1:]):
    sns.barplot(data=moduleProps,x='Network',y=col,hue='Network',palette={comnumber:cmaps[len(comms_filtered_ordered)][comnumber] for comnumber in range(len(comms_filtered_ordered))},ax=ax0_props[N],legend=None)
    ax0_props[N].set_xlabel('')
    N+=1
ax0_props[2].set_title('Topological properties of modules')
ax0_props[2].set_xlabel('Module')
fig.savefig('./fig-abce.pdf')
plt.close()

### Test prokaryotes in modules for niche enrichment
taxaClassification=pd.concat([pd.read_csv('./data/gtdb_taxa_classif.csv').set_index('genus'),pd.read_csv('./data/eukaryotes_classif.tsv',sep='\t').set_index('genus')]) # file generated with the function classifyProkaryotesWithGTDB provided in this script
ecologicalOverrepHM,ecologicalOverrepSignif=testCommunitiesForEcologyOverrepresentation(comms_filtered_ordered,taxaClassification,['soil','freshwater','marine water','plant','animal','sediment','pollution']+['terrestrial viridiplantae','aquatic viridiplantae','metazoans'])
fig,ax=plt.subplots(1,1,figsize=(4,2))
columnsToPlot=['marine water','sediment','freshwater','plant','animal','pollution','soil','aquatic viridiplantae','terrestrial viridiplantae']
sns.heatmap(data=ecologicalOverrepHM[columnsToPlot],cmap='seismic',center=0,ax=ax,annot=ecologicalOverrepSignif[columnsToPlot],fmt='')
plt.savefig('./fig-d.pdf')
plt.close()

##### analyzing negative edges between modules 

## set up figure
fig2 = plt.figure(figsize=(10, 10))
gs2 = fig2.add_gridspec(2,2,width_ratios=[1,1])
ax0 = fig2.add_subplot(gs2[0, 0])
ax1 = fig2.add_subplot(gs2[1, 0])
ax2 = fig2.add_subplot(gs2[:, 1])

## sort association into inter/intra-communities
coocc_neg_signif=coocc[(coocc['p-value']<0.01) & (coocc['correlation']<-0.5)]
interIntra=connectedCommunities(coocc,comms_filtered_ordered,interCommunityOnly=False)
interIntra_signif=interIntra[interIntra['p-value']<0.01]
interIntra_pos=interIntra_signif[(interIntra_signif['correlation']>0.5)]
interIntra_neg=interIntra_signif[(interIntra_signif['correlation']<-0.5)]
#interIntra_neg.to_csv('./interIntra_neg.csv')
barplot_df=[
    {'inter-module':'inter-module', 'type':'negative','Number of edges':len(interIntra_neg[interIntra_neg['inter-module']])},
    {'inter-module':'inter-module', 'type':'positive','Number of edges':len(interIntra_pos[interIntra_pos['inter-module']])},
    {'inter-module':'intra-module', 'type':'negative','Number of edges':len(interIntra_neg[~(interIntra_neg['inter-module'])])},
    {'inter-module':'intra-module', 'type':'positive','Number of edges':len(interIntra_pos[~(interIntra_pos['inter-module'])])}
]
barplot_df=pd.DataFrame(barplot_df)
#barplot_df.to_csv('./interIntra.csv')
sns.barplot(x='Number of edges',y='inter-module',hue='type',data=barplot_df,palette={'positive':'green','negative':'red','not significant':'grey'},ax=ax0)
ax0.set_ylabel('')

## null model shuffling correlations, then reconstructing models and modules
print('running null model shuffling correlations, then reconstructing models and modules...')
counts=getEdgesCounts(coocc_pos_signif,coocc_neg_signif,comms_filtered_ordered)
output_null1=nullModelOnEdgeNumbers(coocc, counts, coocc_pos_signif, coocc_neg_signif,sizeNull=1000)
sns.boxplot(y='Group',x='Proportion of inter-module edges',data=output_null1[output_null1['null_index']!='REAL'],fliersize=0,color='grey',ax=ax1)
sns.swarmplot(y='Group',x='Proportion of inter-module edges',data=output_null1[output_null1['null_index']=='REAL'],size=6,color='black',ax=ax1)

## null model shuffling module compositions
print('running null model shuffling module compositions...')
null_pos,null_neg=nullModelOnModuleCompositions(counts.reset_index(drop=False),coocc_pos_signif, coocc_neg_signif,comms_filtered_ordered,sizeNull=1000)
null_neg4plot=pd.DataFrame(null_neg.drop(columns=['p-val','p-val_2sided']).stack()).reset_index().rename(columns={'level_0':'connected communities','level_1':'index',0:'Number of edges'}).sort_values(by='connected communities')
null_pos4plot=pd.DataFrame(null_pos.stack()).reset_index().rename(columns={'level_0':'connected communities','level_1':'index',0:'Number of edges'}).sort_values(by='connected communities')

## plot
sns.boxplot(y='connected communities',x='Number of edges',ax=ax2,data=null_neg4plot[null_neg4plot['index']!='negative edges'],fliersize=0,color='red')
sns.swarmplot(y='connected communities',x='Number of edges',ax=ax2,data=null_neg4plot[null_neg4plot['index']=='negative edges'],size=6,color='red')
for ind in null_neg.index:
    if null_neg.loc[ind,'p-val']<0.05:
        ax2.text(y=ind,x=null_neg.loc[ind,'negative edges']+150,s='*',va='center',ha='center')
fig2.savefig('./fig-fgh.pdf')
plt.close()


    

