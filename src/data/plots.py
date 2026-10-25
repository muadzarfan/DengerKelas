from .common import ROOT,read_jsonl,event


def plot_distributions():
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={s:read_jsonl(ROOT/f'data/manifests/fleurs_{s}.jsonl') for s in ('train','validation','test')}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5))
    colors={'train':'#2166ac','validation':'#1b9e77','test':'#7570b3'}
    for col,(split,records) in enumerate(rows.items()):
        for ax,field,label in ((axes[0,col],'duration_seconds','Duration (seconds)'),
                               (axes[1,col],'target_token_count','Target tokens including prefix/EOS')):
            values=[r[field] for r in records if field in r]
            ax.hist(values,bins=30,color=colors[split],edgecolor='white',linewidth=.3)
            ax.set_xlabel(label); ax.set_ylabel('Recordings')
            ax.set_title(f'{split}: n={len(records)}')
            ax.grid(axis='y',alpha=.2)
        axes[0,col].axvline(30,color='#b2182b',linestyle='--',label='Single-window limit')
        axes[1,col].text(.98,.97,'Decoder limit: 448 tokens',ha='right',va='top',
            transform=axes[1,col].transAxes,fontsize=7)
        axes[0,col].legend(fontsize=7)
    fig.suptitle('DengerKelas — FLEURS Indonesian integrity audit (all official rows)')
    fig.tight_layout()
    output=ROOT/'results/step3/distributions.png'
    fig.savefig(output,dpi=160,bbox_inches='tight'); plt.close(fig)
    event('audit_plot_created',path='results/step3/distributions.png')
    return output
