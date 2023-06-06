import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.compose import TransformedTargetRegressor
import math
from math import sqrt
from sklearn.pipeline import Pipeline
from joblib import dump, load

import plotly
import plotly.express as px



def lr_customization(desc_file, ep_file, cv_fold, scaler):
    desc_df = desc_file
    desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))
    ep_df = ep_file
    ep_df = ep_df.fillna(ep_df.mean(numeric_only=True))

    # 1. deal with the index of desc and ep df ("ENM")
    # if "ENM" in desc_df:
    #     desc_df = desc_df.set_index("ENM")
    # # else:
    # #     desc_df = desc_df.set_index(desc_df.columns[0])
    # #     desc_df.index.name = 'ENM'
    # if "ENM" in ep_df:
    #     ep_df = ep_df.set_index("ENM")
    # else:
    #     ep_df = ep_df.set_index(ep_df.columns[0])
    #     ep_df.index.name = 'ENM'
    # st.title(desc_df.index)
    # st.title(ep_df.index)
    # st.dataframe(desc_df)

    # 2.1 check the indices of descriptor and endpoint dataset are the same
    if len(desc_df.index) == len(ep_df.index):
        if list(desc_df.index) != list(ep_df.index):
            st.warning('Error! The order or names of samples between descriptor and endpoint dataset are not same.')
            return
    else:
        st.warning('Error! The number of samples between descriptor and endpoint dataset are not same.')
        return

    # TODO if it is necessary to restrict the number of NMs to at least 5
    # if desc_df.shape[0] < 5:
    #     flash('Error! The number of nanomaterials must be greater than four in the modeling dataset')
    #     return

    # 2.2 check if cv_fold is 3, 5, 10 or LOOCV, assign to cv
    #     if cv_fold is not None, assign its value to cv
    if cv_fold == "LOO":
        cv = desc_df.shape[0]
    elif cv_fold == "3-Fold":
        cv = 3
    elif cv_fold == "5-Fold":
        cv = 5
    elif cv_fold == "10-Fold":
        cv = 10

    # check the fold value selected in Cross Validation. It should be smaller than the number of samples.
    if cv > desc_df.shape[0]:
        st.warning(
            "Error! The fold value selected in Cross Validation is lager than the number of samples in descriptor "
            "and endpoint datasets. Please select a smaller fold value.")
        return

    if scaler == "Standard scaler":
        scaler_option = StandardScaler()
    elif scaler == "MinMax scaler":
        scaler_option = MinMaxScaler()

    # 3. create pipeline param, else flash error
    try:
        pipe_lr = Pipeline([("scaler", scaler_option), ("lr", GridSearchCV(LinearRegression(), param_grid={}, cv=cv, scoring='neg_root_mean_squared_error'))])
        # pipe_lr = Pipeline([("scaler", scaler_option), ("lr", GridSearchCV(LinearRegression(), param_grid={}, cv=cv, scoring='r2'))])
    except:
        st.warning("Error！Model can't be created. Please check the parameter and dataset!")
        return

    # 4. train model and get the best one
    x_train = desc_df.iloc[:, :].values
    y_train = ep_df.iloc[:, :].values
    st.title(x_train)
    st.title(y_train)


    lr_model = pipe_lr.fit(x_train, y_train)
    y_pred = lr_model.predict(x_train)

    # Get the coefficient of descriptors.
    # Note: comparing to PLSR, LR model need to reorder nested list from row to column
    coef_list = [list(e) for e in zip(*lr_model.named_steps['lr'].best_estimator_.coef_)]
    coef_df = pd.DataFrame(coef_list, columns=ep_df.columns, index=desc_df.columns)
    coef_df.index.name = "Descriptor"
    # flash(coef_df)
    # coef_df_name = "DescriptorContribution_LinearRegression_" + ep_filename.split('.xlsx')[0] + ".xlsx"
    # coef_df.to_excel(CUSTOM_COEF_OUTPUT_DIR + '/' + coef_df_name)
    coef_df_name = "DescriptorContribution_LinearRegression_Result" + ".xlsx"
    coef_df.to_excel('./' + coef_df_name)

    # Prepare the parameters for descriptor contribution figure
    data_list = list()
    fig_title_list = list()
    labels = list()
    coef_sub_dfs = [coef_df[[col]] for col in coef_df]
    for i in coef_sub_dfs:
        positive_val = i.where(i.gt(0)).count().values[0]
        if positive_val >= 20:
            labels.append(list(i.iloc[:, 0].sort_values(ascending=False).head(20).index))
            data_list.append(list(i.iloc[:, 0].sort_values(ascending=False).head(20).values))
            fig_title_list.extend(list(i.columns.values))
        else:
            labels.append(list(i.iloc[:, 0].sort_values(ascending=False).head(positive_val).index))
            data_list.append(list(i.iloc[:, 0].sort_values(ascending=False).head(positive_val).values))
            fig_title_list.extend(list(i.columns.values))

    # statistics for the best model
    # flash(y_train)
    # flash(y_pred)
    r2 = round(r2_score(y_train.flatten(), y_pred.flatten()), 4)
    mse = mean_squared_error(y_train.flatten(), y_pred.flatten())
    rmse = round(sqrt(mse), 3)
    # Note: the code below is for cross validation result analysis
    y_pred_cv = cross_val_predict(lr_model.named_steps['lr'].best_estimator_, x_train, y_train, cv=cv)
    r2_cv = round(r2_score(y_train.flatten(), y_pred_cv.flatten()), 4)

    # 5. create the regression df (need to check the number of doses)
    if ep_df.shape[1] > 1:
        regression_idx = list()
        idx_name = list(ep_df.index)
        col_name = list(ep_df.columns.values)
        for i in idx_name:
            for j in col_name:
                regression_idx.append(i + "_" + j)
    else:
        regression_idx = ep_df.index

    regression_df = pd.DataFrame({"Experiment": y_train.flatten(), "Prediction": y_pred.flatten()},
                                 index=regression_idx)

    # 6. draw regression figure
    exp_max = math.ceil(regression_df["Experiment"].max())
    exp_min = math.floor(regression_df["Experiment"].min())
    pred_max = math.ceil(regression_df["Prediction"].max())
    pred_min = math.ceil(regression_df["Prediction"].min())

    axis_right = max(exp_max, pred_max)
    axis_left = min(exp_min, pred_min)
    if axis_right >= 0 and axis_left >= 0:
        axis_max = axis_right + (axis_right + axis_left) / 5
        axis_min = axis_left - (axis_right + axis_left) / 5
    elif axis_right >= 0 and axis_left <= 0:
        axis_max = axis_right + axis_right / 5
        axis_min = axis_left + axis_left / 5
    elif axis_right <= 0 and axis_left <= 0:
        axis_max = axis_right - (axis_right + axis_left) / 5
        axis_min = axis_left + (axis_right + axis_left) / 5

    n_colors = regression_df.shape[0]
    # flash(n_colors)
    # flash([n/(n_colors-1) for n in range(n_colors)])
    if n_colors <= 24:
        fig = px.scatter(regression_df, x="Experiment", y="Prediction", color=regression_df.index,
                         color_discrete_sequence=px.colors.qualitative.Light24)
    elif 24 < n_colors <= 26:
        fig = px.scatter(regression_df, x="Experiment", y="Prediction", color=regression_df.index,
                         color_discrete_sequence=px.colors.qualitative.Alphabet)
    else:
        # color_list = [n/(n_colors-1) for n in range(n_colors)]
        # random.shuffle(color_list)
        colors = px.colors.sample_colorscale("Rainbow", [n / (n_colors - 1) for n in range(n_colors)])
        fig = px.scatter(regression_df, x="Experiment", y="Prediction", color=regression_df.index,
                         color_discrete_sequence=colors)
    fig.update_layout(
        shapes=[{'type': 'line', "opacity": 0.3, 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
    fig.update_layout(xaxis_range=[axis_min, axis_max])
    fig.update_layout(yaxis_range=[axis_min, axis_max])
    fig.update_layout(title={'text': '<b>Best LR model <br> from cross validation procedure</b>'})
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title_x=0.5)
    fig.update_traces(marker=dict(size=10, line=dict(width=1.5, color='DarkSlateGrey')), selector=dict(mode='markers'))

    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # output data to the corresponding folder
    # lr_model_name = "Pickle_LinearRegression_" + ep_filename.split('.xlsx')[0] + ".joblib"  # only need to return the name for model pickle download
    # dump(lr_model, CUSTOM_PICKLE_OUTPUT_DIR + '/' + lr_model_name)
    # regression_df_name = "ScatterPlotData_LinearRegression_" + ep_filename.split('.xlsx')[0] + ".xlsx"  # only need to return the name for regression file download
    # regression_df.to_excel(CUSTOM_REG_OUTPUT_DIR + '/' + regression_df_name)
    lr_model_name = "Pickle_LinearRegression_Model" + ".joblib"  # only need to return the name for model pickle download
    dump(lr_model, './' + lr_model_name)
    regression_df_name = "ScatterPlotData_LinearRegression_Result" + ".xlsx"  # only need to return the name for regression file download
    regression_df.to_excel('./' + regression_df_name)

    return r2, rmse, fig, lr_model_name, regression_df_name, coef_df_name, data_list, labels, fig_title_list, r2_cv







with st.sidebar:
    st.image("./img/vinas-toolbox.png")
    st.title("ViNAS-AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "PCA", "AutoML", "Results", "Download"])
    st.info("This application (ViNAS-AutoML) allows you to build automated ML models.")

if os.path.exists("./source_desc_dataset.xlsx"):
    df_desc = pd.read_excel("source_desc_dataset.xlsx", index_col="ENM")
if os.path.exists("./source_ep_dataset.xlsx"):
    df_ep = pd.read_excel("source_ep_dataset.xlsx", index_col="ENM")

if choice == "Upload":
    # upload descriptor dataset
    st.title("Upload descriptor dataset for modelling!")
    file_desc = st.file_uploader("Upload descriptor dataset here!")
    if file_desc:
        df_desc = pd.read_excel(file_desc, index_col="ENM")
        df_desc.to_excel("source_desc_dataset.xlsx")
        st.dataframe(df_desc)
    # upload endpoint dataset
    st.title("Upload endpoint dataset for modelling!")
    file_ep = st.file_uploader("Upload endpoint dataset here!")
    if file_ep:
        df_ep = pd.read_excel(file_ep, index_col="ENM")
        df_ep.to_excel("source_ep_dataset.xlsx")
        st.dataframe(df_ep)


if choice == "Profiling":
    st.title("Automated exploratory data analysis!")
    desc_profile_report = df_desc.profile_report()
    st_profile_report(desc_profile_report)

if choice == "PCA":
    method_option = st.selectbox('Please select PCA method', ('Standard scaler', 'MinMax scaler'))
    pca_button = st.button('PCA analysis')
    if pca_button:
        if method_option == "Standard scaler":
            scaler = StandardScaler()
            scaler.fit(df_desc)
            standard_desc = scaler.transform(df_desc)
            df_standard_desc = pd.DataFrame(standard_desc, index=df_desc.index, columns=df_desc.columns)
            df_standard_desc = df_standard_desc.fillna(df_standard_desc.mean(numeric_only=True))
            # df_standard_desc.to_excel(outputdir + upload_name + "_" + "stdscaler.xlsx")

            # PCA_2D_dataset
            df_normal = df_standard_desc
            pca_res2d = PCA(2).fit_transform(df_normal.values)
            df_pca2d = pd.DataFrame(pca_res2d, index=df_normal.index, columns=["PC1", "PC2"])
            # df_pca2d.to_excel(outputdir + upload_name + "_" + "stdPCA2D.xlsx")
            st.title("PCA_2D_Dataset")
            st.dataframe(df_pca2d)
            # PCA_3D_dataset
            pca_res3d = PCA(3).fit_transform(df_normal.values)
            df_pca3d = pd.DataFrame(pca_res3d, index=df_normal.index, columns=["PC1", "PC2", "PC3"])
            # df_pca3d.to_excel(outputdir + upload_name + "_" + "stdPCA3D.xlsx")
            st.title("PCA_3D_Dataset")
            st.dataframe(df_pca3d)
            # draw 2D and 3D PCA figures
            n_colors_2d = df_pca2d.shape[0]
            colors_2d = px.colors.sample_colorscale("Rainbow", [n / (n_colors_2d - 1) for n in range(n_colors_2d)])
            n_colors_3d = df_pca3d.shape[0]
            colors_3d = px.colors.sample_colorscale("Rainbow", [n / (n_colors_3d - 1) for n in range(n_colors_3d)])
            fig1 = px.scatter(df_pca2d, x="PC1", y="PC2",
                              color=df_pca2d.index,  title="upload_name_2d", color_discrete_sequence=colors_2d)
            fig1.update_traces(marker=dict(size=10, line=dict(width=1.5, color='DarkSlateGrey')), selector=dict(mode='markers'))

            fig2 = px.scatter_3d(df_pca3d, x="PC1", y="PC2", z="PC3", color=df_pca3d.index,
                                 title="upload_name_3d", color_discrete_sequence=colors_3d)
            fig2.update_traces(marker=dict(size=8, line=dict(width=1.5, color='DarkSlateGrey')), selector=dict(mode='markers'))

            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

        if method_option == "MinMax scaler":
            scaler = MinMaxScaler()
            scaler.fit(df_desc)
            minmax_desc = scaler.transform(df_desc)

            df_minmax_desc = pd.DataFrame(minmax_desc, index=df_desc.index, columns=df_desc.columns)
            df_minmax_desc = df_minmax_desc.fillna(df_minmax_desc.mean(numeric_only=True))
            # df_minmax_desc.to_excel(outputdir + upload_name + "_" + "mmscaler.xlsx")

            # PCA_2D_dataset
            df_normal = df_minmax_desc
            pca_res2d = PCA(2).fit_transform(df_normal.values)
            df_pca2d = pd.DataFrame(pca_res2d, index=df_normal.index, columns=["PC1", "PC2"])
            # df_pca2d.to_excel(outputdir + upload_name + "_" + "mmPCA2D.xlsx")
            st.title("PCA_2D_Dataset")
            st.dataframe(df_pca2d)
            # PCA_3D_dataset
            pca_res3d = PCA(3).fit_transform(df_normal.values)
            df_pca3d = pd.DataFrame(pca_res3d, index=df_normal.index, columns=["PC1", "PC2", "PC3"])
            # df_pca3d.to_excel(outputdir + upload_name + "_" + "mmPCA3D.xlsx")
            st.title("PCA_3D_Dataset")
            st.dataframe(df_pca3d)
            # draw 2D and 3D PCA figures
            n_colors_2d = df_pca2d.shape[0]
            colors_2d = px.colors.sample_colorscale("Rainbow", [n / (n_colors_2d - 1) for n in range(n_colors_2d)])
            n_colors_3d = df_pca3d.shape[0]
            colors_3d = px.colors.sample_colorscale("Rainbow", [n / (n_colors_3d - 1) for n in range(n_colors_3d)])

            fig1 = px.scatter(df_pca2d, x="PC1", y="PC2",
                              color=df_pca2d.index,  title="upload_name_2d", color_discrete_sequence=colors_2d)
            fig1.update_traces(marker=dict(size=10, line=dict(width=1.5, color='DarkSlateGrey')), selector=dict(mode='markers'))
            fig2 = px.scatter_3d(df_pca3d, x="PC1", y="PC2", z="PC3", color=df_pca3d.index,
                                 title="upload_name_3d", color_discrete_sequence=colors_3d)
            fig2.update_traces(marker=dict(size=8, line=dict(width=1.5, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)




if choice == "AutoML":
    ml_method_option = st.selectbox('Please select ML method', ('Linear Regression', 'Partial Least Square Regression'))
    if ml_method_option == 'Linear Regression':
        scaler_method_option = st.selectbox('Please select scaler method', ('Standard scaler', 'MinMax scaler'))
        cv_method_option = st.selectbox('Please select cross validation method', ('3-Fold', '5-Fold', '10-Fold', 'LOO'))
        modeling_button = st.button('Submit for Modeling')
        if modeling_button:
            st.title("modeling......")
            st.dataframe(df_desc)
            st.dataframe(df_ep)
            lr_results = list(lr_customization(df_desc, df_ep, cv_method_option, scaler_method_option))
            st.title(lr_results[0])
            st.title(lr_results[1])
            st.plotly_chart(lr_results[2])
    if ml_method_option == 'Partial Least Square Regression':
        pass


