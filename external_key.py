"""Independent fit-only key discovery, followed by causal history shrinkage."""
import numpy as np
import pandas as pd


class AutoKeyMemory:
    def configuration(self):
        return {'hypothesis':'HYP-106','strategy':'training_only_greedy_entity_key_memory',
            'max_columns':6,'max_cardinality_fraction':.9,'min_discovery_coverage':.2,
            'max_target_within_std_ratio':.5,'step_relative_gain':.01,
            'discovery_query':'last_training_period_features_only',
            'score':'coverage_div_mean_rows_per_key_period',
            'history':'strictly_past_exponential_half_life_one',
            'confidence':'min(.9,n/(n+1))*exp(-.25*max(0,gap-1))',
            'missing_key':'no_match','fit_only':True,'query_statistics':False}

    @staticmethod
    def _tokens(X,columns):
        result=[]
        for c in columns:
            s=X[c]
            if pd.api.types.is_numeric_dtype(s):
                part=pd.to_numeric(s,errors='coerce').map(lambda x:None if not np.isfinite(x) else 'N:'+format(float(x),'.17g'))
            else:part=s.map(lambda x:None if pd.isna(x) else 'S:'+str(x))
            result.append(part.tolist())
        return [tuple(v) if all(x is not None for x in v) else None for v in zip(*result)]

    @staticmethod
    def _years(X):
        years=pd.to_numeric(X['Year_Factor'],errors='raise').to_numpy(float)
        if not np.isfinite(years).all():raise ValueError('Finite explicit periods required')
        return years

    def fit(self,X,y):
        y=np.asarray(y,float)
        if not isinstance(X,pd.DataFrame) or not len(X) or y.shape!=(len(X),) or not np.isfinite(y).all():
            raise ValueError('Finite aligned training inputs required')
        years=self._years(X);latest=years.max();query=years==latest;prefix=~query
        self.columns_=[];self.records_={};self.steps_=[];self.raw_columns_=list(X.columns)
        candidates=sorted(c for c in X if c!='Year_Factor' and str(c).lower() not in ['id','site_eui','target','label']
                          and 2<=X[c].nunique(dropna=True)<.9*len(X))
        best=0.
        if prefix.any():
            for _ in range(6):
                winner=None
                for c in candidates:
                    if c in self.columns_:continue
                    cols=self.columns_+[c];tokens=self._tokens(X,cols)
                    past={tokens[i] for i in np.flatnonzero(prefix) if tokens[i] is not None}
                    coverage=np.mean([tokens[i] in past if tokens[i] is not None else False for i in np.flatnonzero(query)])
                    moments={}
                    for key,year in zip(tokens,years):
                        if key is not None:moments[(key,year)]=moments.get((key,year),0)+1
                    repeat=np.mean(list(moments.values())) if moments else np.inf
                    score=float(coverage/repeat)
                    if score>best*1.01 and (winner is None or score>winner[0]):winner=(score,c,float(coverage),float(repeat))
                if winner is None:break
                best,c,coverage,repeat=winner;self.columns_.append(c)
                self.steps_.append({'column':c,'score':best,'coverage':coverage,'rows_per_key_period':repeat})
        ratio=None;reason='no_eligible_key'
        if self.columns_:
            tokens=self._tokens(X,self.columns_);grouped={}
            for key,year,target in zip(tokens,years,y):
                if key is not None:grouped.setdefault(key,[]).append((float(year),float(target)))
            stds=[np.std([t for year,t in rows],ddof=1) for rows in grouped.values() if len(rows)>1]
            ratio=float(np.mean(stds)/max(np.std(y,ddof=1),1e-12)) if stds else 1.
            if self.steps_[-1]['coverage']>=.2 and ratio<=.5:
                self.records_={key:sorted(rows) for key,rows in grouped.items()};reason='accepted'
            else:reason='coverage_or_within_variance_veto'
        self.audit_={'configuration':self.configuration(),'selected_columns':self.columns_.copy(),
            'steps':self.steps_,'within_std_ratio':ratio,'reason':reason,'profiles':len(self.records_),
            'fit_rows':len(X),'model_learner_fits':0,'row_deletion':False}
        return self

    def lookup(self,X):
        if list(X.columns)!=self.raw_columns_:raise ValueError('Schema mismatch')
        means=np.zeros(len(X));weights=np.zeros(len(X))
        if not self.records_:return means,weights
        for i,(key,year) in enumerate(zip(self._tokens(X,self.columns_),self._years(X))):
            past=[r for r in self.records_.get(key,[]) if r[0]<year]
            if not past:continue
            p=np.array(past,float);last=p[:,0].max();w=.5**(last-p[:,0])
            means[i]=np.average(p[:,1],weights=w)
            weights[i]=min(.9,len(p)/(len(p)+1))*np.exp(-.25*max(0,year-last-1))
        return means,weights

    def predict_with_fallback(self,X,fallback):
        base=np.asarray(fallback,float)
        if base.shape!=(len(X),) or not np.isfinite(base).all():raise ValueError('Finite aligned fallback required')
        means,weights=self.lookup(X)
        return weights*means+(1-weights)*base
