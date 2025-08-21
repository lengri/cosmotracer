import matplotlib.pyplot as plt
import numpy as np 

class SegmentPicker:
    
    def __init__(self, x, y, additional_break_ids = []):
        
        self.x = x 
        self.y = y
        self.breakpoints = additional_break_ids 
        
        self.seg_start_indices = []
        self.seg_stop_indices = []
        
        
    def pick_segments(self, xlabel=r"$x$ [m]"):
        
        fg, ax = plt.subplots(1, 1)
        ax.plot(self.x, self.y, picker=True, c="black", zorder=1)
        for bp in self.breakpoints:
            ax.scatter(self.x[bp], self.y[bp], c="blue", zorder=10)
        ax.set_title(r"Pick segments to use in R$^2$ optimisation")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$z$ [m]")
        
        # Define click event handler
        def _on_pick(event):
            ind = event.ind[0]  # Get index of clicked point

            if ind in self.seg_start_indices or ind in self.seg_stop_indices:
                return  # Skip if already selected
            if len(self.seg_start_indices) == len(self.seg_stop_indices):
                self.seg_start_indices.append(ind)
            else:
                self.seg_stop_indices.append(ind)
                
            # after every click, plot all the points in red
            ax.scatter(self.x[self.seg_start_indices], self.y[self.seg_start_indices], c="red")
            ax.scatter(self.x[self.seg_stop_indices], self.y[self.seg_stop_indices], c="red")
            
            # and plot the allowed segments in red
            for istart, istop in zip(self.seg_start_indices, self.seg_stop_indices):
                ax.plot(self.x[istart:istop+1], self.y[istart:istop+1], c="red")
                
            plt.draw()
        
        fg.canvas.mpl_connect("pick_event", _on_pick)
        plt.show()
        
    def create_segments(self):
        
        # First, we sort the start/stop arrays to make sure that they are in ascending id order.
        
        if len(self.seg_start_indices) == 0:
            return (None, None) 
        
        id_sort = np.argsort(self.seg_start_indices)
        
        self.breakpoints = np.sort(self.breakpoints)
        ids_start = np.array(self.seg_start_indices)[id_sort]
        ids_stop = np.array(self.seg_stop_indices)[id_sort]
        
        # ignore breakpoints that are not within any of the segments
        breakpoints_keep = np.zeros_like(self.breakpoints, dtype=bool)
        for i, id_break in enumerate(self.breakpoints):
            for id0, id1 in zip(ids_start, ids_stop):
                if id_break > id0 and id_break < id1:
                    breakpoints_keep[i] = 1
        
        breakpoints = self.breakpoints[breakpoints_keep]        

        # iterate over both arrays and see where a breakpoint has to be added
        n_insert = 0
        for bp in breakpoints:
            # find the indices in which to insert the breakpoint
            for i, (i0, i1) in enumerate(zip(ids_start, ids_stop)):

                if bp > i0 and bp < i1:
                    # insert here
                    ids_stop = np.insert(ids_stop, i+n_insert, bp)
                    ids_start = np.insert(ids_start, i+1+n_insert, bp)
                    n_insert += 1
                    break

        # Last step: return nested lists of all the segmented x and y values.
        # Note: we want the endpoints to be included in the returned array.
        # So we increase the value by one.
        x_out = []
        y_out = []
        ids_used = []
        all_ids = np.arange(0, len(self.x), 1)
        
        for i0, i1 in zip(ids_start, ids_stop):
            try:
                x_out.append(self.x[i0:i1+1])
                y_out.append(self.y[i0:i1+1])
                ids_used.append(all_ids[i0:i1+1])
            except IndexError: # include last element
                x_out.append(self.x[i0:])
                y_out.append(self.y[i0:])
                ids_used.append(all_ids[i0:])
                
        return (ids_used, x_out, y_out)


def pick_points(x, y, xlabel="", ylabel=""):
    
    fg, ax = plt.subplots(1, 1)
    ax.plot(x, y, picker=True, c="black", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Pick points with your cursor")
    
    points = []
    
    def _on_pick(event):
        ind = event.ind[0]
        if ind in points:
            return  # Skip if already selected
        points.append(ind)
        ax.scatter(x[points], y[points], c="blue")
        plt.draw()
        
    fg.canvas.mpl_connect("pick_event", _on_pick)
    plt.show()
    
    return (points, x[points], y[points])

if __name__ == "__main__":
    
    x = np.linspace(0, 50, 100)
    y = x**2 + np.random.normal(0, 5, size=x.shape)
    
    p = SegmentPicker(x, y, breakpoints=[50, 95])
    p.pick_segments()
    xseg, yseg = p.create_segments()     
    
    plt.plot(x, y)
    for xx, yy in zip(xseg, yseg):
        plt.plot(xx, yy)
    plt.show()