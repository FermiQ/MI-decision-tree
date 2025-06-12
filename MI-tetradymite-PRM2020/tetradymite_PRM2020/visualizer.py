import plotly.graph_objects as go
import ipywidgets as widgets
from jupyter_jsmol import JsmolView
import numpy as np
from IPython.display import display, Markdown, FileLink
import os
from scipy.spatial import ConvexHull
import copy
import pandas as pd

class Visualizer:
    def __init__(self, df, sisso, classes):

        self.sisso = sisso
        self.df = df
        self.class0 = str(classes[0])
        self.class1 = str(classes[1])
        self.marker_size = 7
        self.marker_symbol_cls0 = 'circle'
        self.marker_symbol_cls1 = 'square'
        self.symbols = [
            'circle',
            'square',
            'triangle-up',
            'triangle-down',
            'circle-cross',
            'circle-x'
        ]
        self.font_size = 12
        self.cross_size = 15
        self.hullsline_width = 1
        self.clsline_width = 1
        self.font_families = ['Source Sans Pro',
                              'Helvetica',
                              'Open Sans',
                              'Times New Roman',
                              'Arial',
                              'Verdana',
                              'Courier New',
                              'Comic Sans MS',
                              ]
        self.line_styles = ["dash",
                            "solid",
                            "dot",
                            "longdash",
                            "dashdot",
                            "longdashdot"]
        self.gradient_list = ['Blue to red',
                              'Blue to green',
                              'Green to red',
                              'Grey scale',
                              'Purple scale',
                              'Turquoise scale']
        self.bg_color = 'rgba(229,236,246, 0.5)'
        self.color_cls1 = "#EB8273"
        self.color_cls0 = "rgb(138, 147, 248)"
        self.color_hull0 = 'Grey'
        self.color_hull1 = 'Grey'
        self.color_line = 'Black'

        self.total_features = sisso.n_dim
        self.features = list(reversed([str(str(feat.expr)) for feat in sisso.models[sisso.n_dim - 1][0].feats]))
        self.coefficients = list(reversed(sisso.models[sisso.n_dim - 1][0].coefs[0][:-1]))
        self.intercept = sisso.models[sisso.n_dim - 1][0].coefs[0][-1]
        self.df_cls0 = df['Classification'] == 0
        self.df_cls1 = df['Classification'] == 1
        self.compounds_list = df['Compound'].to_list()
        self.bg_toggle = True
        self.npoints_cls0 = len(self.df_cls0)
        self.npoints_cls1 = len(self.df_cls1)
        self.symbols_cls0 = [self.marker_symbol_cls0] * self.npoints_cls0
        self.symbols_cls1 = [self.marker_symbol_cls1] * self.npoints_cls1
        self.sizes_cls0 = [self.marker_size] * self.npoints_cls0
        self.sizes_cls1 = [self.marker_size] * self.npoints_cls1
        self.colors_cls0 = [self.color_cls0] * self.npoints_cls0
        self.colors_cls1 = [self.color_cls1] * self.npoints_cls1

        self.fig = go.FigureWidget()
        self.viewer_l = JsmolView()
        self.viewer_r = JsmolView()
        self.instantiate_widgets()
        x_cls0 = df[self.features[0]][self.df_cls0]
        y_cls0 = df[self.features[1]][self.df_cls0]
        x_cls1 = df[self.features[0]][self.df_cls1]
        y_cls1 = df[self.features[1]][self.df_cls1]
        line_x, line_y = self.f_x(self.features[0], self.features[1])

        # custom_cls0 = np.dstack((self.target_train[self.df_cls0],
        #                          self.target_predict[self.df_cls0]))[0]
        # custom_cls1 = np.dstack((self.target_train[self.df_cls1],
        #                          self.target_predict[self.df_cls1]))[0]

        self.fig.add_trace(
            (
                go.Scatter(
                    mode='markers',
                    x=x_cls0,
                    y=y_cls0,
                    # customdata=self.cls0,
                    text=df[self.df_cls0]['Compound'].to_numpy(),
                    hovertemplate=
                    r"<b>%{text}</b><br><br>" +
                    "x axis: %{x:,.2f}<br>" +
                    "y axis: %{y:,.2f}<br>",
                    # "ΔE reference:  %{customdata[0]:,.4f}<br>" +
                    # "ΔE predicted:  %{customdata[1]:,.4f}<br>"
                    name='Class 0:<br>' + str(self.class0),
                    marker=dict(color=self.colors_cls0),
                )
            ))
        self.fig.add_trace(
            (
                go.Scatter(
                    mode='markers',
                    x=x_cls1,
                    y=y_cls1,
                    # customdata=custom_cls1,
                    text=df[self.df_cls1]['Compound'].to_numpy(),
                    hovertemplate=
                    r"<b>%{text}</b><br><br>" +
                    "x axis: %{x:,.2f}<br>" +
                    "y axis: %{y:,.2f}<br>",
                    # "ΔE reference:  %{customdata[0]:,.4f}<br>" +
                    # "ΔE predicted:  %{customdata[1]:,.4f}<br>",
                    name='Class 1:<br>' + str(self.class1),
                    marker=dict(color=self.colors_cls1),
                )
            ))

        try:
            hullx_cls0, hully_cls0, hullx_cls1, hully_cls1 = self.make_hull(self.features[0], self.features[1])

            self.fig.add_trace(
                go.Scatter(
                    x=hullx_cls0,
                    y=hully_cls0,
                    line=dict(color=self.color_hull0, width=1, dash=self.line_styles[0]),
                    name=r'Convex' + '<br>' + 'hull 0',
                    visible=False
                ),
            )
            self.fig.add_trace(
                go.Scatter(
                    x=hullx_cls1,
                    y=hully_cls1,
                    line=dict(color=self.color_hull1, width=1, dash=self.line_styles[0]),
                    name=r'Convex' + '<br>' + 'hull 1',
                    visible=False
                ),
            )
        except:
            self.fig.add_trace(
                go.Scatter(
                    visible=False
                ),
            )
            self.fig.add_trace(
                go.Scatter(
                    visible=False
                ),
            )

        self.fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                line=dict(color=self.color_line, width=1, dash='solid'),
                name=r'Classification' + '<br>' + 'line',
                visible=False
            ),

        )
        x_min = min(min(x_cls0), min(x_cls1))
        y_min = min(min(y_cls0), min(y_cls1))
        x_max = max(max(x_cls0), max(x_cls1))
        y_max = max(max(y_cls0), max(y_cls1))
        x_delta = 0.05 * abs(x_max - x_min)
        y_delta = 0.05 * abs(y_max - y_min)
        self.fig.update_layout(
            plot_bgcolor=self.bg_color,
            font=dict(
                size=int(self.font_size),
                family=self.font_families[0]
            ),
            xaxis_title='$D_1$',
            yaxis_title='$D_2$',
            xaxis_range=[x_min - x_delta, x_max + x_delta],
            yaxis_range=[y_min - y_delta, y_max + y_delta],
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            width=800,
            height=400,
            margin=dict(
                l=50,
                r=50,
                b=70,
                t=20,
                pad=4
            ),
        )
        self.fig.update_xaxes(ticks="outside", tickwidth=1, ticklen=10, linewidth=1, linecolor='black')
        self.fig.update_yaxes(ticks="outside", tickwidth=1, ticklen=10, linewidth=1, linecolor='black')

        self.scatter_cls0 = self.fig.data[0]
        self.scatter_cls1 = self.fig.data[1]
        self.scatter_hull0 = self.fig.data[2]
        self.scatter_hull1 = self.fig.data[3]
        self.scatter_clsline = self.fig.data[4]
        
        if self.total_features == 2:
            self.scatter_hull0.visible = True
            self.scatter_hull1.visible = True
            self.scatter_clsline.visible = True
        else:
            self.widg_hullslinewidth.disabled = True
            self.widg_hullslinestyle.disabled = True

        self.update_markers()

    def f_x(self, feat_x, feat_y):
        idx_x = self.features.index(feat_x)
        idx_y = self.features.index(feat_y)
        line_x = np.linspace(self.df[feat_x].min(), self.df[feat_x].max(), 1000)

        # Gives the classifications line
        if self.widg_featx.value == self.widg_featy.value:
            return line_x, line_x
        else:
            line_y = -line_x * self.coefficients[idx_x] / self.coefficients[idx_y] - self.intercept / self.coefficients[
                idx_y]
            return line_x, line_y

    def make_hull(self, feat_x, feat_y):

        points_0 = self.df[self.df_cls0][[feat_x, feat_y]].to_numpy()
        delta_0 = max(points_0[:, 0]) - min(points_0[:, 0])
        delta_1 = max(points_0[:, 1]) - min(points_0[:, 1])
        exp_1 = int(np.log10(delta_0 / delta_1))
        exp_0 = int(np.log10(delta_1 / delta_0))
        if exp_1 > 6:
            points_0[:, 1] = points_0[:, 1] * 10 ** exp_1
        if exp_0 > 6:
            points_0[:, 0] = points_0[:, 0] * 10 ** exp_0
        hull_cls0 = ConvexHull(points_0)
        vertexes_cls0 = self.df[self.df_cls0][[feat_x, feat_y]].to_numpy()[hull_cls0.vertices]

        points_1 = self.df[self.df_cls1][[feat_x, feat_y]].to_numpy()
        delta_0 = max(points_1[:, 0]) - min(points_1[:, 0])
        delta_1 = max(points_1[:, 1]) - min(points_1[:, 1])
        exp_1 = int(np.log10(delta_0 / delta_1))
        exp_0 = int(np.log10(delta_1 / delta_0))
        if exp_1 > 6:
            points_1[:, 1] = points_1[:, 1] * 10 ** exp_1
        if exp_0 > 6:
            points_1[:, 0] = points_1[:, 0] * 10 ** exp_0
        hull_cls1 = ConvexHull(points_1)
        vertexes_cls1 = self.df[self.df_cls1][[feat_x, feat_y]].to_numpy()[hull_cls1.vertices]

        x_hullvx_cls0 = vertexes_cls0[:, 0]
        y_hullvx_cls0 = vertexes_cls0[:, 1]
        x_hullvx_cls1 = vertexes_cls1[:, 0]
        y_hullvx_cls1 = vertexes_cls1[:, 1]
        n_intervals = 100

        xhull_cls0 = np.array([x_hullvx_cls0[0]])
        yhull_cls0 = np.array([y_hullvx_cls0[0]])
        for xy in zip(x_hullvx_cls0, y_hullvx_cls0):
            xhull_cls0 = np.concatenate([xhull_cls0, np.linspace(xhull_cls0[-1], xy[0], n_intervals)])
            yhull_cls0 = np.concatenate([yhull_cls0, np.linspace(yhull_cls0[-1], xy[1], n_intervals)])
        xhull_cls0 = np.concatenate([xhull_cls0, np.linspace(xhull_cls0[-1], x_hullvx_cls0[0], n_intervals)])
        yhull_cls0 = np.concatenate([yhull_cls0, np.linspace(yhull_cls0[-1], y_hullvx_cls0[0], n_intervals)])

        xhull_cls1 = np.array([x_hullvx_cls1[0]])
        yhull_cls1 = np.array([y_hullvx_cls1[0]])
        for xy in zip(x_hullvx_cls1, y_hullvx_cls1):
            xhull_cls1 = np.concatenate([xhull_cls1, np.linspace(xhull_cls1[-1], xy[0], n_intervals)])
            yhull_cls1 = np.concatenate([yhull_cls1, np.linspace(yhull_cls1[-1], xy[1], n_intervals)])
        xhull_cls1 = np.concatenate(
            [xhull_cls1, np.linspace(xhull_cls1[-1], x_hullvx_cls1[0], n_intervals)])
        yhull_cls1 = np.concatenate(
            [yhull_cls1, np.linspace(yhull_cls1[-1], y_hullvx_cls1[0], n_intervals)])

        return xhull_cls0, yhull_cls0, xhull_cls1, yhull_cls1

    def update_markers(self):
        # Markers size and symbol are updated simultaneously
        with self.fig.batch_update():
            self.scatter_cls0.marker.size = self.sizes_cls0
            self.scatter_cls1.marker.size = self.sizes_cls1
            self.scatter_cls0.marker.symbol = self.symbols_cls0
            self.scatter_cls1.marker.symbol = self.symbols_cls1
            self.scatter_cls0.marker.color = self.colors_cls0
            self.scatter_cls1.marker.color = self.colors_cls1

    def set_markers_size(self, feature='Default size'):
        # Defines the size of the markers based on the input feature.
        # In case of default feature all markers have the same size.
        # Points marked with x/cross are set with a specific size

        if feature == 'Default size':

            sizes_cls0 = [self.marker_size] * self.npoints_cls0
            sizes_cls1 = [self.marker_size] * self.npoints_cls1
            symbols_cls0 = self.symbols_cls0
            symbols_cls1 = self.symbols_cls1

            try:
                point = symbols_cls0.index('x')
                sizes_cls0[point] = self.cross_size
            except:
                try:
                    point = symbols_cls1.index('x')
                    sizes_cls1[point] = self.cross_size
                except:
                    pass
            try:
                point = symbols_cls0.index('cross')
                sizes_cls0[point] = self.cross_size
            except:
                try:
                    point = symbols_cls1.index('cross')
                    sizes_cls1[point] = self.cross_size
                except:
                    pass

            self.sizes_cls0 = sizes_cls0
            self.sizes_cls1 = sizes_cls1
        else:
            min_value = min(min(self.df.loc[self.df_cls0][feature]),
                            min(self.df.loc[self.df_cls1][feature]))
            max_value = max(max(self.df.loc[self.df_cls0][feature]),
                            max(self.df.loc[self.df_cls1][feature]))
            coeff = 2 * self.marker_size / (max_value - min_value)
            sizes_cls0 = self.marker_size / 2 + coeff * self.df.loc[self.df_cls0][
                feature]
            sizes_cls1 = self.marker_size / 2 + coeff * self.df.loc[self.df_cls1][
                feature]
            self.sizes_cls0 = sizes_cls0
            self.sizes_cls1 = sizes_cls1

    def make_colors(self, feature, gradient):

        if feature == 'Default color':

            self.colors_cls0 = [self.color_cls0] * self.npoints_cls0
            self.colors_cls1 = [self.color_cls1] * self.npoints_cls1

        else:

            min_value = self.df[feature].min()
            max_value = self.df[feature].max()
            shade_cls0 = 0.7 * (self.df.loc[self.df_cls0][feature].to_numpy() - min_value) / \
                         (max_value-min_value)
            shade_cls1 = 0.7 * (self.df.loc[self.df_cls1][feature].to_numpy() - min_value) / \
                         (max_value-min_value)

            if gradient == 'Grey scale':
                for i, e in enumerate(shade_cls0):
                    value = 255*(0.7-e)
                    string = 'rgb('+str(value)+","+str(value)+","+str(value)+')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255*(0.7-e)
                    string = 'rgb('+str(value)+","+str(value)+","+str(value)+')'
                    self.colors_cls1[i] = string

            if gradient == 'Purple scale':
                for i, e in enumerate(shade_cls0):
                    value = 255 * (0.7 - e)
                    string = 'rgb(' + str(value) + "," + str(0) + "," + str(value) + ')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255 * (0.7 - e)
                    string = 'rgb(' + str(value) + "," + str(0) + "," + str(value) + ')'
                    self.colors_cls1[i] = string

            if gradient == 'Turquoise scale':
                for i, e in enumerate(shade_cls0):
                    value = 255 * (0.7 - e)
                    string = 'rgb(' + str(0) + "," + str(value) + "," + str(value) + ')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255 * (0.7 - e)
                    string = 'rgb(' + str(0) + "," + str(value) + "," + str(value) + ')'
                    self.colors_cls1[i] = string

            shade_cls0 = 0.7 * (
                    self.df.loc[self.df_cls0][feature].to_numpy() - min_value) / (max_value - min_value)
            shade_cls1 = 0.7 * (
                    self.df.loc[self.df_cls1][feature].to_numpy() - min_value) / (max_value - min_value)
            if gradient == 'Blue to green':
                for i, e in enumerate(shade_cls0):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(0) + "," + str(value) + "," + str(value2) + ')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(0) + "," + str(value) + "," + str(value2) + ')'
                    self.colors_cls1[i] = string

            if gradient == 'Blue to red':
                for i, e in enumerate(shade_cls0):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(value) + "," + str(0) + "," + str(value2) + ')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(value) + "," + str(0) + "," + str(value2) + ')'
                    self.colors_cls1[i] = string

            if gradient == 'Green to red':
                for i, e in enumerate(shade_cls0):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(value) + "," + str(value2) + "," + str(0) + ')'
                    self.colors_cls0[i] = string
                for i, e in enumerate(shade_cls1):
                    value = 255 * e
                    value2 = 255 - value
                    string = 'rgb(' + str(value) + "," + str(value2) + "," + str(0) + ')'
                    self.colors_cls1[i] = string

    def handle_xfeat_change(self, change):
        # changes the feature plotted on the x-axis
        # separating line is modified accordingly
        feat_x = change.new
        feat_y = self.widg_featy.value

        self.scatter_cls0['x'] = self.df.loc[self.df_cls0][feat_x].to_numpy()
        self.scatter_cls1['x'] = self.df.loc[self.df_cls1][feat_x].to_numpy()

        line_x, line_y = self.f_x(feat_x, feat_y)
        min_x = min(min(self.scatter_cls0['x']), min(self.scatter_cls1['x']))
        max_x = max(max(self.scatter_cls0['x']), max(self.scatter_cls1['x']))
        min_delta = 0.05 * abs(max_x - min_x)

        with self.fig.batch_update():
            self.scatter_clsline['x'] = line_x
            self.scatter_clsline['y'] = line_y
            self.fig.layout['xaxis'].range = [min_x - min_delta, max_x + min_delta]
            if feat_x == feat_y:
                self.scatter_hull0.visible = False
                self.scatter_hull1.visible = False
                self.scatter_clsline.visible = False
            else:
                self.scatter_hull0.visible = True
                self.scatter_hull1.visible = True
                self.scatter_clsline.visible = True
                hullx_cls0, hully_cls0, hullx_cls1, hully_cls1 = self.make_hull(feat_x, feat_y)
                self.scatter_hull0['x'] = hullx_cls0
                self.scatter_hull0['y'] = hully_cls0
                self.scatter_hull1['x'] = hullx_cls1
                self.scatter_hull1['y'] = hully_cls1

        self.widg_feat_labelx.value = r'$D_1 = ' + str(feat_x) + "$"

    def handle_yfeat_change(self, change):
        # changes the feature plotted on the x-axis
        # separating line is modified accordingly
        feat_x = self.widg_featx.value
        feat_y = change.new

        self.scatter_cls0['y'] = self.df.loc[self.df_cls0][feat_y].to_numpy()
        self.scatter_cls1['y'] = self.df.loc[self.df_cls1][feat_y].to_numpy()

        line_x, line_y = self.f_x(feat_x, feat_y)
        min_y = min(min(self.scatter_cls0['y']), min(self.scatter_cls1['y']))
        max_y = max(max(self.scatter_cls0['y']), max(self.scatter_cls1['y']))
        min_delta = 0.05 * abs(max_y - min_y)

        with self.fig.batch_update():
            self.scatter_clsline['x'] = line_x
            self.scatter_clsline['y'] = line_y
            self.fig.layout['yaxis'].range = [min_y - min_delta, max_y + min_delta]
            if feat_x == feat_y:
                self.scatter_hull0.visible = False
                self.scatter_hull1.visible = False
                self.scatter_clsline.visible = False
            else:
                self.scatter_hull0.visible = True
                self.scatter_hull1.visible = True
                self.scatter_clsline.visible = True
                hullx_cls0, hully_cls0, hullx_cls1, hully_cls1 = self.make_hull(feat_x, feat_y)
                self.scatter_hull0['x'] = hullx_cls0
                self.scatter_hull0['y'] = hully_cls0
                self.scatter_hull1['x'] = hullx_cls1
                self.scatter_hull1['y'] = hully_cls1

        self.widg_feat_labely.value = r'$D_2 = ' + str(feat_y) + "$"

    def handle_markerfeat_change(self, change):
        self.set_markers_size(feature=change.new)
        self.update_markers()

    def handle_colorfeat_change(self, change):
        if change.new == 'Default color':
            self.widg_gradient.layout.visibility = 'hidden'
            self.colors_cls0 = [self.color_cls0] * self.npoints_cls0
            self.colors_cls1 = [self.color_cls1] * self.npoints_cls1
        else:
            self.widg_gradient.layout.visibility = 'visible'
            self.make_colors(feature=change.new, gradient=self.widg_gradient.value)
        self.update_markers()

    def handle_gradient_change(self, change):
        self.make_colors(feature=self.widg_featcolor.value, gradient=change.new)
        self.update_markers()

    def updatecolor_button_clicked(self, button):

        if self.widg_featcolor.value == 'Default color':
            try:
                self.scatter_cls0.update(marker=dict(color=self.widg_color_cls0.value))
                self.color_cls0 = self.widg_color_cls0.value
                self.colors_cls0 = self.npoints_cls0 * [self.color_cls0]
            except:
                pass
            try:
                self.scatter_cls1.update(marker=dict(color=self.widg_color_cls1.value))
                self.color_cls0 = self.widg_color_cls0.value
                self.colors_cls0 = self.npoints_cls0 * [self.color_cls0]
            except:
                pass

            if self.bg_toggle:
                try:
                    self.fig.update_layout(plot_bgcolor=self.widg_bgcolor.value)
                    self.bg_color = self.widg_bgcolor.value
                except:
                    pass
        try:
            self.scatter_clsline.update(line=dict(color=self.widg_color_line.value))
            self.color_line = self.widg_color_hull0.value
        except:
            pass
        try:
            self.scatter_hull0.update(line=dict(color=self.widg_color_hull0.value))
            self.color_hull0 = self.widg_color_hull0.value
        except:
            pass
        try:
            self.scatter_hull1.update(line=dict(color=self.widg_color_hull1.value))
            self.color_hull1 = self.widg_color_hull1.value
        except:
            pass

    def handle_fontfamily_change(self, change):

        self.fig.update_layout(
            font=dict(family=change.new)
        )

    def handle_fontsize_change(self, change):

        self.fig.update_layout(
            font=dict(size=change.new)
        )

    def handle_markersize_change(self, change):

        self.marker_size = int(change.new)
        self.set_markers_size(feature=self.widg_featmarker.value)
        self.update_markers()

    def handle_crossize_change(self, change):

        self.cross_size = int(change.new)
        self.set_markers_size(feature=self.widg_featmarker.value)
        self.update_markers()

    def handle_hullslinewidth_change(self, change):

        self.hullsline_width = change.new
        with self.fig.batch_update():
            self.scatter_hull0.line.width = change.new
            self.scatter_hull1.line.width = change.new

    def handle_hullslinestyle_change(self, change):

        with self.fig.batch_update():
            self.scatter_hull0.line.dash = change.new
            self.scatter_hull1.line.dash = change.new

    def handle_clslinewidth_change(self, change):

        self.clsline_width = change.new
        with self.fig.batch_update():
            self.scatter_clsline.line.width = change.new

    def handle_clslinestyle_change(self, change):

        with self.fig.batch_update():
            self.scatter_clsline.line.dash = change.new

    def handle_markersymbol_cls0_change(self, change):

        for i, e in enumerate(self.symbols_cls0):
            if e == self.marker_symbol_cls0:
                self.symbols_cls0[i] = change.new
        self.marker_symbol_cls0 = change.new
        self.update_markers()

    def handle_markersymbol_cls1_change(self, change):

        for i, e in enumerate(self.symbols_cls1):
            if e == self.marker_symbol_cls1:
                self.symbols_cls1[i] = change.new
        self.marker_symbol_cls1 = change.new
        self.update_markers()

    def bgtoggle_button_clicked(self, button):

        if self.bg_toggle:
            self.bg_toggle = False
            self.fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(gridcolor='rgb(229,236,246)', showgrid=True, zeroline=False),
                yaxis=dict(gridcolor='rgb(229,236,246)', showgrid=True, zeroline=False),
            )
        else:
            self.bg_toggle = True
            self.fig.update_layout(
                plot_bgcolor=self.widg_bgcolor.value,
                xaxis=dict(gridcolor='white'),
                yaxis=dict(gridcolor='white')
            )

    def print_button_clicked(self, button):

        self.widg_print_out.clear_output()
        text = "A download link will appear soon."
        with self.widg_print_out:
            display(Markdown(text))
        path = "./data/tetradymite_PRM2020/plots/"
        try:
            os.mkdir(path)
        except:
            pass
        file_name = self.widg_plot_name.value + '.' + self.widg_plot_format.value
        self.fig.write_image(path + file_name, scale=self.widg_scale.value)
        self.widg_print_out.clear_output()
        with self.widg_print_out:
            local_file = FileLink(path + file_name, result_html_prefix="Click here to download: ")
            display(local_file)

    def reset_button_clicked(self, button):

        self.symbols_cls0 = [self.marker_symbol_cls0] * self.npoints_cls0
        self.symbols_cls1 = [self.marker_symbol_cls1] * self.npoints_cls1
        self.set_markers_size(self.widg_featmarker.value)
        self.update_markers()

    def plotappearance_button_clicked(self, button):
        if self.widg_box_utils.layout.visibility == 'visible':
            self.widg_box_utils.layout.visibility = 'hidden'
            for i in range(490, -1, -1):
                self.widg_box_viewers.layout.top = str(i) + 'px'
            self.widg_box_utils.layout.bottom = '0px'
        else:
            for i in range(491):
                self.widg_box_viewers.layout.top = str(i) + 'px'
            self.widg_box_utils.layout.bottom = '400px'
            self.widg_box_utils.layout.visibility = 'visible'

    def handle_checkbox_l(self, change):
        if change.new:
            self.widg_checkbox_r.value = False
        else:
            self.widg_checkbox_r.value = True

    def handle_checkbox_r(self, change):
        if change.new:
            self.widg_checkbox_l.value = False
        else:
            self.widg_checkbox_l.value = True

    def view_structure_cls0_l(self, formula):
        self.viewer_l.script("load data/tetradymite_PRM2020/structures/" + formula + ".xyz")

    def view_structure_cls0_r(self, formula):
        self.viewer_r.script("load data/tetradymite_PRM2020/structures/" + formula + ".xyz")

    def view_structure_cls1_l(self, formula):
        self.viewer_l.script("load data/tetradymite_PRM2020/structures/" + formula + ".xyz")

    def view_structure_cls1_r(self, formula):
        self.viewer_r.script("load data/tetradymite_PRM2020/structures/" + formula + ".xyz")

    def display_button_l_clicked(self, button):

        # Actions are performed only if the string inserted in the text widget corresponds to an existing compound
        if self.widg_compound_text_l.value in self.compounds_list:

            self.viewer_l.script("load data/tetradymite_PRM2020/structures/" + self.widg_compound_text_l.value + ".xyz")

            symbols_cls0 = self.symbols_cls0
            symbols_cls1 = self.symbols_cls1

            try:
                point = symbols_cls0.index('x')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('x')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass

            try:
                point = np.where(self.scatter_cls0['text'] == self.widg_compound_text_l.value)[0][0]
                symbols_cls0[point] = 'x'
            except:
                try:
                    point = np.where(self.scatter_cls1['text'] == self.widg_compound_text_l.value)[0][0]
                    symbols_cls1[point] = 'x'
                except:
                    pass

            self.symbols_cls1 = symbols_cls1
            self.symbols_cls0 = symbols_cls0
            self.set_markers_size(feature=self.widg_featmarker.value)
            self.update_markers()

    def display_button_r_clicked(self, button):

        # Actions are performed only if the string inserted in the text widget corresponds to an existing compound
        if self.widg_compound_text_r.value in self.compounds_list:

            self.viewer_r.script("load data/tetradymite_PRM2020/structures/" + self.widg_compound_text_r.value + ".xyz")

            symbols_cls0 = self.symbols_cls0
            symbols_cls1 = self.symbols_cls1

            try:
                point = symbols_cls0.index('cross')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('cross')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass
            try:
                point = np.where(self.scatter_cls0['text'] == self.widg_compound_text_r.value)[0][0]
                symbols_cls0[point] = 'cross'
            except:
                try:
                    point = np.where(self.scatter_cls1['text'] == self.widg_compound_text_r.value)[0][0]
                    symbols_cls1[point] = 'cross'
                except:
                    pass

            self.symbols_cls0 = symbols_cls0
            self.symbols_cls1 = symbols_cls1
            self.set_markers_size(feature=self.widg_featmarker.value)
            self.update_markers()

    def update_point_cls0(self, trace, points, selector):
        # changes the points labeled with a cross on the map.
        if not points.point_inds:
            return

        symbols_cls0 = self.symbols_cls0
        symbols_cls1 = self.symbols_cls1

        # The element previously marked with x/cross is marked with circle as default value
        if self.widg_checkbox_l.value:
            try:
                point = symbols_cls0.index('x')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('x')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass
        if self.widg_checkbox_r.value:
            try:
                point = symbols_cls0.index('cross')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('cross')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass

        if self.widg_checkbox_l.value:
            symbols_cls0[points.point_inds[0]] = 'x'
        if self.widg_checkbox_r.value:
            symbols_cls0[points.point_inds[0]] = 'cross'

        self.symbols_cls0 = symbols_cls0
        self.symbols_cls1 = symbols_cls1
        self.set_markers_size(feature=self.widg_featmarker.value)
        self.update_markers()

        formula = trace['text'][points.point_inds[0]]
        if self.widg_checkbox_l.value:
            self.widg_compound_text_l.value = formula
            self.view_structure_cls0_l(formula)
        if self.widg_checkbox_r.value:
            self.widg_compound_text_r.value = formula
            self.view_structure_cls0_r(formula)

    def update_point_cls1(self, trace, points, selector):
        if not points.point_inds:
            return

        symbols_cls0 = self.symbols_cls0
        symbols_cls1 = self.symbols_cls1

        # The element previously marked with x/cross is marked with circle as default value
        if self.widg_checkbox_l.value:
            try:
                point = symbols_cls0.index('x')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('x')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass
        if self.widg_checkbox_r.value:
            try:
                point = symbols_cls0.index('cross')
                symbols_cls0[point] = self.marker_symbol_cls0
            except:
                try:
                    point = symbols_cls1.index('cross')
                    symbols_cls1[point] = self.marker_symbol_cls1
                except:
                    pass

        if self.widg_checkbox_l.value:
            symbols_cls1[points.point_inds[0]] = 'x'
        if self.widg_checkbox_r.value:
            symbols_cls1[points.point_inds[0]] = 'cross'

        self.symbols_cls0 = symbols_cls0
        self.symbols_cls1 = symbols_cls1
        self.set_markers_size(feature=self.widg_featmarker.value)
        self.update_markers()

        formula = trace['text'][points.point_inds[0]]
        if self.widg_checkbox_l.value:
            self.widg_compound_text_l.value = formula
            self.view_structure_cls1_l(formula)
        if self.widg_checkbox_r.value:
            self.widg_compound_text_r.value = formula
            self.view_structure_cls1_r(formula)

    def show(self):

        self.widg_featx.observe(self.handle_xfeat_change, names='value')
        self.widg_featy.observe(self.handle_yfeat_change, names='value')
        self.widg_featmarker.observe(self.handle_markerfeat_change, names='value')
        self.widg_featcolor.observe(self.handle_colorfeat_change, names='value')
        self.widg_gradient.observe(self.handle_gradient_change, names='value')
        self.widg_checkbox_l.observe(self.handle_checkbox_l, names='value')
        self.widg_checkbox_r.observe(self.handle_checkbox_r, names='value')
        self.widg_display_button_l.on_click(self.display_button_l_clicked)
        self.widg_display_button_r.on_click(self.display_button_r_clicked)
        self.widg_updatecolor_button.on_click(self.updatecolor_button_clicked)
        self.widg_reset_button.on_click(self.reset_button_clicked)
        self.widg_print_button.on_click(self.print_button_clicked)
        self.widg_bgtoggle_button.on_click(self.bgtoggle_button_clicked)
        self.scatter_cls0.on_click(self.update_point_cls0)
        self.scatter_cls1.on_click(self.update_point_cls1)
        self.widg_markersize.observe(self.handle_markersize_change, names='value')
        self.widg_crosssize.observe(self.handle_crossize_change, names='value')
        self.widg_hullslinewidth.observe(self.handle_hullslinewidth_change, names='value')
        self.widg_hullslinestyle.observe(self.handle_hullslinestyle_change, names='value')
        self.widg_clslinewidth.observe(self.handle_clslinewidth_change, names='value')
        self.widg_clslinestyle.observe(self.handle_clslinestyle_change, names='value')
        self.widg_fontfamily.observe(self.handle_fontfamily_change, names='value')
        self.widg_fontsize.observe(self.handle_fontsize_change, names='value')
        self.widg_plotutils_button.on_click(self.plotappearance_button_clicked)
        self.widg_markersymbol_cls1.observe(self.handle_markersymbol_cls1_change, names='value')
        self.widg_markersymbol_cls0.observe(self.handle_markersymbol_cls0_change, names='value')

        self.output_l.layout = widgets.Layout(width="400px", height='350px')
        self.output_r.layout = widgets.Layout(width="400px", height='350px')

        with self.output_l:
            display(self.viewer_l)
        with self.output_r:
            display(self.viewer_r)

        self.widg_box_utils.layout.visibility = 'hidden'
        self.widg_gradient.layout.visibility = 'hidden'

        self.box_feat.layout.height = '150px'
        self.box_feat.layout.top = '30px'
        self.widg_plotutils_button.layout.left = '50px'


        self.widg_box_utils.layout.border = 'dashed 1px'
        self.widg_box_utils.right = '100px'
        self.widg_box_utils.layout.max_width = '700px'

        container = widgets.VBox([
                                  self.box_feat, self.fig,
                                  self.widg_plotutils_button,
                                  self.widg_box_viewers,
                                  self.widg_box_utils
                                  ])

        display(container)

    def instantiate_widgets(self):

        self.widg_featx = widgets.Dropdown(
            description='x-axis',
            options=self.features,
            value=self.features[0]
        )
        self.widg_featy = widgets.Dropdown(
            description='y-axis',
            options=self.features,
            value=self.features[1]
        )
        self.widg_featmarker = widgets.Dropdown(
            description="Marker",
            options=['Default size'] + self.features,
            value='Default size',
        )
        self.widg_featcolor = widgets.Dropdown(
            description='Color',
            options=['Default color'] + self.features,
            value='Default color'
        )
        self.widg_gradient = widgets.Dropdown(
            description='-gradient',
            options=self.gradient_list,
            value='Grey scale',
            layout=widgets.Layout(width='150px', right='20px')
        )
        self.widg_compound_text_l = widgets.Combobox(
            placeholder='...',
            description='Compound:',
            options=self.compounds_list,
            disabled=False,
            layout=widgets.Layout(width='200px')
        )
        self.widg_compound_text_r = widgets.Combobox(
            placeholder='...',
            description='Compound:',
            options=self.compounds_list,
            disabled=False,
            layout=widgets.Layout(width='200px')
        )
        self.widg_display_button_l = widgets.Button(
            description="Display",
            layout=widgets.Layout(width='100px')
        )
        self.widg_display_button_r = widgets.Button(
            description="Display",
            layout=widgets.Layout(width='100px')
        )
        self.widg_checkbox_l = widgets.Checkbox(
            value=True,
            indent=False,
            layout=widgets.Layout(width='50px')
        )
        self.widg_checkbox_r = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='50px'),
        )
        self.widg_markersize = widgets.BoundedIntText(
            placeholder=str(self.marker_size),
            description='Marker size',
            value=str(self.marker_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_crosssize = widgets.BoundedIntText(
            placeholder=str(self.cross_size),
            description='Cross size',
            value=str(self.cross_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_fontsize = widgets.BoundedIntText(
            placeholder=str(self.font_size),
            description='Font size',
            value=str(self.font_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_hullslinewidth = widgets.BoundedIntText(
            placeholder=str(self.hullsline_width),
            description='Hulls width',
            value=str(self.hullsline_width),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_hullslinestyle = widgets.Dropdown(
            options=self.line_styles,
            description='Hulls style',
            value=self.line_styles[0],
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_clslinewidth = widgets.BoundedIntText(
            placeholder=str(self.clsline_width),
            description='Line width',
            value=str(self.clsline_width),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_clslinestyle = widgets.Dropdown(
            options=self.line_styles,
            description='Line style',
            value='solid',
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_fontfamily = widgets.Dropdown(
            options=self.font_families,
            description='Font family',
            value=self.font_families[0],
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_bgcolor = widgets.Text(
            placeholder=str(self.bg_color),
            description='Background',
            value=str(self.bg_color),
            layout=widgets.Layout(left='30px', width='200px'),

        )
        self.widg_color_cls0 = widgets.Text(
            placeholder=str(self.color_cls0),
            description='Color 0',
            value=str(self.color_cls0),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_color_cls1 = widgets.Text(
            placeholder=str(self.color_cls1),
            description='Color 1',
            value=str(self.color_cls1),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_color_hull0 = widgets.Text(
            placeholder=str(self.color_hull0),
            description='Hull 0',
            value=str(self.color_hull0),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_color_hull1 = widgets.Text(
            placeholder=str(self.color_hull1),
            description='Hull 1',
            value=str(self.color_hull1),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_color_line = widgets.Text(
            placeholder=str(self.color_line),
            description='Color line',
            value=str(self.color_line),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_markersymbol_cls0 = widgets.Dropdown(
            description='Symbol 0',
            options=self.symbols,
            value=self.marker_symbol_cls0,
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_markersymbol_cls1 = widgets.Dropdown(
            description='Symbol 1',
            options=self.symbols,
            value=self.marker_symbol_cls1,
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_bgtoggle_button = widgets.Button(
            description='Toggle on/off background',
            layout=widgets.Layout(left='50px', width='200px'),
        )
        self.widg_updatecolor_button = widgets.Button(
            description='Update colors',
            layout=widgets.Layout(left='50px', width='200px')
        )
        self.widg_reset_button = widgets.Button(
            description='Reset symbols',
            layout=widgets.Layout(left='50px',width='200px')
        )
        self.widg_plot_name = widgets.Text(
            placeholder='plot',
            value='plot',
            description='Name',
            layout=widgets.Layout(width='300px')
        )
        self.widg_plot_format = widgets.Text(
            placeholder='png',
            value='png',
            description='Format',
            layout=widgets.Layout(width='150px')
        )
        self.widg_scale = widgets.Text(
            placeholder='1',
            value='1',
            description="Scale",
            layout=widgets.Layout(width='150px')
        )
        self.widg_print_button = widgets.Button(
            description='Print',
            layout=widgets.Layout(left='50px', width='600px')
        )
        self.widg_print_out = widgets.Output(
            layout=widgets.Layout(left='50px', width='600px')
        )
        self.widg_feat_labelx = widgets.Label(
            value=r"$D_1 = " + str(self.widg_featx.value) + "$",
            layout=widgets.Layout(left='55px', width='600px', top='5px')
        )
        self.widg_feat_labely = widgets.Label(
            value=r"$D_2 = " + str(self.widg_featy.value) + "$",
            layout=widgets.Layout(left='55px', width='600px')
        )
        self.widg_printdescription = widgets.Label(
            value="Click 'Print' to export the plot in the desired format.",
            layout=widgets.Layout(left='50px', width='640px')
        )
        self.widg_printdescription2 = widgets.Label(
            value="The resolution of the image can be increased by increasing the 'Scale' value.",
            layout=widgets.Layout(left='50px', width='640px')
        )
        self.widg_featuredescription = widgets.Label(
            value="The dropdown menus select the features to visualize."
        )
        self.widg_description = widgets.Label(
            value='Tick the box next to the cross symbols in order to choose which windows visualizes the next '
                  'structure selected in the map above.'
        )
        self.widg_colordescription = widgets.Label(
            value='Colors in the boxes below can be written as a text string, i.e. red, '
                  'green,...,  or in a rgb/a, hex format. ',
            layout=widgets.Layout(left='50px', width='640px')

        )
        self.widg_colordescription2 = widgets.Label(
            value="After modifying a specific field, click on the 'Update colors' button to display the changes in "
                  "the plot.",
            layout=widgets.Layout(left='50px', width='640px')
        )
        self.widg_plotutils_button = widgets.Button(
            description='Toggle on/off the plot appearance utils',
            layout=widgets.Layout(width='600px')
        )
        self.widg_box_utils = widgets.VBox([widgets.HBox([self.widg_markersize, self.widg_crosssize,
                                                          self.widg_fontsize]),
                                            widgets.HBox([self.widg_hullslinewidth, self.widg_hullslinestyle,
                                                          self.widg_fontfamily]),
                                            widgets.HBox([self.widg_clslinewidth, self.widg_clslinestyle]),
                                            widgets.HBox([self.widg_markersymbol_cls0, self.widg_markersymbol_cls1]),
                                            self.widg_colordescription, self.widg_colordescription2,
                                            widgets.HBox([self.widg_color_cls0, self.widg_color_cls1, self.widg_bgcolor]),
                                            widgets.HBox([self.widg_color_line, self.widg_color_hull0, self.widg_color_hull1]),
                                            widgets.HBox([self.widg_bgtoggle_button,self.widg_updatecolor_button,
                                                          self.widg_reset_button]),
                                            self.widg_printdescription, self.widg_printdescription2,
                                            widgets.HBox([self.widg_plot_name, self.widg_plot_format, self.widg_scale]),
                                            self.widg_print_button, self.widg_print_out,
                                            ])

        file1 = open("./assets/tetradymite_PRM2020/cross.png", "rb")
        image1 = file1.read()
        self.widg_img1 = widgets.Image(
            value=image1,
            format='png',
            width=30,
            height=30,
        )
        file2 = open("./assets/tetradymite_PRM2020/cross2.png", "rb")
        image2 = file2.read()
        self.widg_img2 = widgets.Image(
            value=image2,
            format='png',
            width=30,
            height=30,
        )
        self.output_l = widgets.Output()
        self.output_r = widgets.Output()

        self.box_feat = widgets.VBox([widgets.HBox([widgets.VBox([self.widg_featx, self.widg_featy]),
                                                     widgets.VBox([self.widg_featmarker,
                                                                   widgets.HBox([self.widg_featcolor, self.widg_gradient])
                                                                   ])]),
                                      self.widg_feat_labelx,
                                      self.widg_feat_labely
                                      ])

        self.widg_box_viewers = widgets.VBox([self.widg_description, widgets.HBox([
            widgets.VBox([
                widgets.HBox([self.widg_compound_text_l, self.widg_display_button_l,
                              self.widg_img1, self.widg_checkbox_l]),
                self.output_l]),
            widgets.VBox(
                [widgets.HBox([self.widg_compound_text_r, self.widg_display_button_r,
                               self.widg_img2, self.widg_checkbox_r]),
                 self.output_r])
        ])])
