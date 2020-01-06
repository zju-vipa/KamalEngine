from visdom import Visdom
import json

class VisdomPlotter(object):
    """ Visualizer
    """
    def __init__(self, port='13579', env='main'):
        self.cur_win = {}
        self.env = env 
        self.visdom = Visdom(port=port, env=env)

    def add_scalar(self, win, x, y, opts=None, trace_name=None):
        """ Draw line
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        default_opts = {'title': win}
        if opts is not None:
            default_opts.update(opts)
        update = 'append' if win is not None else None
        self.visdom.line(X=x, Y=y, opts=default_opts, win=win, env=self.env, update=update, name=trace_name)

    def add_image(self, win, img, opts=None):
        """ vis image in visdom
        """
        default_opts = dict(title=win)
        if opts is not None:
            default_opts.update(opts)
        self.visdom.image(img=img, win=win, opts=default_opts, env=self.env)

    def add_table(self, win, tbl, opts=None):
        tbl_str = "<table width=\"100%\"> "
        tbl_str += "<tr> \
                 <th>[Key]</th> \
                 <th>[Value]</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str += "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>" % (k, v)
        tbl_str += "</table>"

        default_opts = {'title': win}
        if opts is not None:
            default_opts.update(opts)
        self.visdom.text(tbl_str, win=win, env=self.env, opts=default_opts)