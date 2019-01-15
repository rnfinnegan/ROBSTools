# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:35:21 2016

@author: robbie
"""


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def LabelledBoxPlot(df, val, sep, title, **kwargs):
    fig = plt.figure(figsize=(12,7))
    ax=fig.add_subplot(111)

    hfont = {'fontname':'monospace', 'size':'small'}

    sns.boxplot(x=val, y=sep, data=df, whis=np.inf, ax=ax)
    #sns.stripplot(x=val, y=sep, data=df, jitter=True, size=3, color=".3", linewidth=0, ax=ax)
    sns.despine()
    ax.set_xlim(0,1.01)
    ax.grid()

    sepList = [y.get_text() for y in ax.get_yticklabels()]
    ylocs = ax.get_yticks()
    for i in ylocs:
        mu = np.nanmean(df.loc[df[sep]==sepList[i],val].values)
        std = np.nanstd(df.loc[df[sep]==sepList[i],val].values)
        ax.text(1.02,i, '{0:7.3f} '.format(mu)+r'$\pm$'+'{0:7.3f}'.format(std), **hfont)
    ax.text(1.02, -1, 'Mean  Std. Dev.')
    fig.suptitle(title)
    fig.subplots_adjust(left=0.20, right=0.8)
    return fig,ax

def LabelledBoxPlotGeneral(df, val, sep, title, **kwargs):
    fig = plt.figure(figsize=(12,7))
    ax=fig.add_subplot(111)

    hfont = {'fontname':'monospace', 'size':'small'}

    sns.boxplot(x=val, y=sep, data=df, whis=np.inf, ax=ax)
    #sns.stripplot(x=val, y=sep, data=df, jitter=True, size=3, color=".3", linewidth=0, ax=ax)
    sns.despine()
    ax.set_xlim(df[val].min(), 1.1*df[val].max())
    ax.grid()

    sepList = [y.get_text() for y in ax.get_yticklabels()]
    ylocs = ax.get_yticks()
    xext = (ax.get_xlim()[0])+1.02*np.abs(ax.get_xlim()[1]-ax.get_xlim()[0])
    for i in ylocs:
        mu = np.nanmean(df.loc[df[sep]==sepList[i],val].values)
        std = np.nanstd(df.loc[df[sep]==sepList[i],val].values)
        ax.text(xext,i, '{0:7.3f} '.format(mu)+r'$\pm$'+'{0:7.3f}'.format(std), **hfont)
    ax.text(xext, -1, 'Mean   Std. Dev.')
    fig.suptitle(title)
    fig.subplots_adjust(left=0.20, right=0.8)
    return fig

def addOverlay(fig, df, val, sep, name):
    ax=fig.axes[0]
    sns.swarmplot(x=val, y=sep, data=df, color='black', edgecolor='black', ax=ax)
    xext = (ax.get_xlim()[0])+1.25*np.abs(ax.get_xlim()[1]-ax.get_xlim()[0])
    ax.text(xext, -1, name)

    sepList = [y.get_text() for y in ax.get_yticklabels()]
    ylocs = ax.get_yticks()
    for i in ylocs:
        pval = df.loc[df[sep]==sepList[i], val].values
        ax.text(xext,i, r'${0:7.3f}$'.format(pval[0]))
    fig.subplots_adjust(right=0.75)

    return fig
