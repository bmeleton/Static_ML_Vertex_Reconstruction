"""
Defines :class:`Vertex`, :class:`Intersection`, :class:`Line`, :class:`Mytree`, and :class:`Event` classes

Defines functions these classes depend on, and other functions useful for point cloud analysis

:class:`Event`: Contains event info such as its number, file locations, point cloud data, Mytree data, and its event classification

:class:`Mytree`: Contains line and vertex data for the event. Has many methods for line and vertex analysis/manipulation

:class:`Line`: Contains data for a specific line in a tree including its number, fit properties, its vertices, and points associated

:class:`Vertex`: Contains data for a specific vertex in a tree including its id, location, and associated lines

:class:`Intersection`: Acts as a preliminary vertex. Is more accuratly the point of closest aproach to a pair of lines


Classes were initially created and used in Classes_v1.py
moved here so they can be used in more programs and have a single definition for modification 
"""
import os, re, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ROOT

''' '''

# region Classes
# **************************************** Vertex class *****************************************************************************
class Vertex:
    # region:Vertex
    '''
    id, point, radius, lines, tree 
    '''
    _id_counter = 1

    def __init__(self, pt, rad, lines:list = None, tree=None):
        self.id = Vertex._id_counter
        Vertex._id_counter += 1
        
        self.point = np.array(pt)
        self.radius = float(rad)
        self.lines = set() # references to Line objects
        if lines != None:
            for line in lines:
                self.lines.add(line)
        self.tree = tree
        if not tree == None:
            tree.vertices.add(self)
    def __repr__(self): return f"Vertex {self.id}"

    def clear(self): # removes references from vertex. Doesn't delete it from the tree.
        if self.lines != None:
            for Lin in self.lines:
                if self is Lin.v2:
                    Lin.v2=None
                elif self is Lin.v1:
                    Lin.v1=Lin.v2
                    Lin.v2=None
                else: print(f"  {self} not found in Line {Lin.lineNum} for removal")
        self.lines = None
        self.id = 0 # signal the vertex is cleared
        self.tree = None
        return

    def add_line(self, Lin): # Tries to add itself to the line, then add the line to itself
        if Lin.v1 == None and Lin.v2 == None:
            Lin.v1 = self # if lines empty, just assign to v1

        elif Lin.v1 is self or Lin.v2 is self:
            print(f"  Vertex {self.id} already in Line {Lin.lineNum}") # v already in line, do nothing

        elif Lin.v2 == None and Lin.v1 != None:
            Lin.v2 = self # v1 is full and not our v, so put v in v2

        else: # implies v1 and v2 both full and aren't our v. remove line from out v
            print(f"  Line {Lin.lineNum} already has 2 verts, couldn't add to vert {self.id}")
            return self
        
        if Lin not in self.lines:
            self.lines.add(Lin)
        else: print(f"  Line {Lin.lineNum} already in vert {self.id}")

        return self
    
    def remove_line(self, Lin): # Tries to remove line from self, then remove self from line
        if Lin not in self.lines:
            print(f"  Line {Lin.lineNum} not in vertex {self.id} for removal")
        else: self.lines.discard(Lin)

        if Lin.v1 is self:
            Lin.v1 = Lin.v2
            Lin.v2 = None
        elif Lin.v2 is self:
            Lin.v2 = None
        else:
            print(f"  Couldn't find vertex {self.id} in Line {Lin.lineNum} to remove it.")
        return

    def refit(self): # refits the vertex point and radius to all of its lines. returns how much it changed? -------------------- TODO implament changed
        if self.lines is None or not self.lines: # if vertex has no lines
            print(f"  couldn't refit {self}, found no lines")
            return None
        elif len(self.lines) == 1:
            print(f"  couldn't refit {self}, only has one line ------ unexpectedish")
            return None
        
        change = None
        #lines = list(self.lines) # need fixed iteration ?
        points=list() # list of all closest points, np arrays aren't hashable
        base:Line
        line:Line
        for i, base in enumerate(self.lines):
            # condition maybe idk
            for j, line in enumerate(self.lines):
                if j<=i: continue # avoid double counting
                curr_pts = closest_points(base,line) # finds points on base/line closest to line/base
                avg_pt = np.mean(curr_pts, axis=0) # compute average point of those two points
                points.append(avg_pt) # add average point to set for later
        
        new_point = np.mean(points, axis=0)
        print(f"  {self} : {self.point} -> {new_point}")
        self.point = new_point
        # refit radius too

        return change

# **************************************** Intersection class **************************************************************
class Intersection:
    # region:Intersection
    '''
    id, point, line1, point1, line2, point2, vertex, tree, score
    '''
    _id_counter = 1
    def __init__(self, pt, line1, point1, line2, point2, vertex = None, tree=None): # can add an associated vetex
        self.id = Intersection._id_counter
        Intersection._id_counter += 1
        self.point = np.array(pt)

        self.line1 = line1
        self.point1 = point1 # point on line 1 closest to line 2
        self.line2 = line2
        self.point2 = point2 # point on line 2 closest to line 1
        self.vertex = vertex # intersection not added to vertex, doesn't care
        self.tree = tree
        line1.intersections.add(self)
        line2.intersections.add(self)
        self.score = self.get_score() 
        if tree != None:
            tree.intersections.add(self)

    def __repr__(self): return f"Inter: {self.id}"
    #def __repr__(self): return f"Inter: {self.id}, {self.line1}, {self.line2}"

    def clear(self): # removes all references from and to Intersection. Doesn't delete from tree
        self.id = 0
        self.line2.remove_intersection(self)
        self.line1.remove_intersection(self)
        self.vertex = None
        self.tree = None
        return

    def add_vertex(self, v):
        if self.vertex != None:
            print(f"  changing Intersection {self.id} vertex from vertex {self.vertex.id} to {v.id}")
        self.vertex = v
        return self

    def get_score(self):
        perp = 1.0-np.dot(self.line1.b,self.line2.b) # slopes < 1 -> dot <= 1 -> 1.0-dot gives score of orthogonality

        pos = 1/min(self.line1.position, self.line2.position)

        score = perp
        self.score = score
        return score

# **************************************** Line class ************************************************************** TODO consider line density and consider distance of points from line
class Line: # lineNum, numOfPts, a, b, start, end, bSphere, points, v1, v2, tree, intersections
    # region:Line
    """
    Line object that has points, vertices, a tree, intersections, and more!
    Attributes:
        lineNum: number assigned to line. Used for id
        numOfPts: number of points the line has
        a: centroid of lines data
        b: vector direction of line
        start: start of the line, closest to 0,0,0
        end: end of the line, furthest to 0,0,0
        points: pd Dataframe of points in the line
        v1: first Vertex
        v2: second Vertex
    """
    # endregion

    def __init__(self, line_num:float, num_of_pts:int, a:np.ndarray, b:np.ndarray, start:np.ndarray, end:np.ndarray, points:pd.DataFrame, v1:Vertex=None, v2:Vertex=None, tree=None):
        self.lineNum = float(line_num)
        self.numOfPts = int(num_of_pts)
        self.a = np.array(a)
        slope = np.array(b)
        slope /= np.linalg.norm(slope)
        if np.linalg.norm(end) < np.linalg.norm(start):
            # if line is backwards, flip it and its slope
            self.start, self.end = end, start 
            self.b = -slope
        else:
            self.b = slope
            self.start =np.array(start)
            self.end = np.array(end)
        #self.bSphere = np_cartesian_to_spherical(b) # (r, theta, phi)
        self.set_points(points)
        self.v1 = v1
        self.v2 = v2
        self.tree = tree
        self.intersections=set()
        self.min_z = self.points['z'].min()
        self.position = 1
        print(f"line: {self.lineNum}\ndensity: {self.density}")
        if v1 != None:
            v1.add_line(self)
        if v2 != None:
            v2.add_line(self)
        if tree != None:
            tree.add_line(self)
        
    #def __repr__(self): return f"Line number: {self.lineNum} \nPoints: \n{self.points} \n"
    def __repr__(self): return f"Line number: {self.lineNum}"

    def clear(self): # removes references to and from line. Doesn't delete it from the tree
        self.remove_vertices()
        self.numOfPts = 0 # signals line has been cleared
        self.tree = None
        if self.intersections != None:
            for inter in self.intersections:
                if inter.line1 is self:
                    inter.line1 = inter.line2
                    inter.line2 = None
                elif inter.line2 is self:
                    inter.line2 = None
                else: print(f"  couldn't find Line {self.lineNum} in Intersection {inter.id} to clear it")
        self.intersections = None
        return None
    
    def Print(self, points=False):
        print(f"  =line num:    {self.lineNum}=")
        print(f"num of points:  {self.numOfPts}")
        #print(f"centroid, a:    {self.a}")
        print(f"slope, b:       {self.b}")
        print(f"start -> end:   {self.start} -> {self.end}")

        return

    @property
    def density(self):
        self.length = np.linalg.norm(self.end-self.start)
        return self.numOfPts / self.length

    def set_points(self, points):
        """takes in points df and adds info like dist to curr line"""
        pd.options.mode.chained_assignment = None  # Suppresses SettingWithCopyWarning globally
        pts = points[['x','y','z']].values
        points['line_dist'] = np.linalg.norm( np.cross(self.b, (pts - self.a) ) , axis=1) / np.linalg.norm(self.b)

        tmp = np.array(pts-self.a)
        norms = np.linalg.norm(tmp, axis=1)
        tmp = ((tmp/norms[:, np.newaxis])-self.b)
        points['slope_dist'] = np.linalg.norm(tmp, axis=1)
        # print(points)
        # print(tmp)
        # print(norms)

        self.points=points
        return points
    def add_vertex(self, v): # Tries to add vertex to self, then add self to vertex
        v.add_line(self) # lazy but should work 
        return self
    def remove_vertex(self, v): # Tries to remove self from vert, then remove vert from self
        v.remove_line(self) # again lazy but should work
        # if not self in v.lines:
        #     print(f"  Line {self.lineNum} not in vertex")
        # else:
        # v.lines.discard(self)
        # if v is self.v1:
        #     self.v1 = self.v2
        #     self.v2 = None
        # elif v is self.v2:
        #     self.v2 = None
        # else:
        #     print(f"  Couldn't find vert in Line {self.lineNum} to remove it.")
        return
    def remove_vertices(self): # uses remove_vertex on v2 then v1
        if self.v2 != None: # removes v2 frist because I wanna try to maintain that v2 should only exist if there is a v1
            self.remove_vertex(self.v2)
        if self.v1 != None:
            self.remove_vertex(self.v1)
        return self

    def add_intersection(self, intersection): # tries to put line in intersection, then intersection into line
        if intersection.line1 == None and intersection.line2 == None: # if inter empty, add line
            intersection.line1 = self
        elif intersection.line1 is self or intersection.line2 is self: # line already in inter
            print(f"  Line {self.lineNum} already in Intersection {intersection.id}")
        elif intersection.line1 != None and intersection.line2 == None: # line1 full, try line2
            intersection.line2 = self
        else: 
            print(f"  Intersection {intersection.id} already full, couldn't add Line {self.lineNum}") 
            return self
        
        if intersection not in self.intersections:
            self.intersections.add(intersection)
        else: print(f"  Line {self.lineNum} already in Intersection {intersection.id}")
        return self
    def remove_intersection(self, intersection):
        if intersection not in self.intersections:
            print(f"  Intersection {intersection.id} not in Line {self.lineNum}")
        else: self.intersections.discard(intersection)

        if intersection.line1 is self:
            intersection.line1 = intersection.line2
            intersection.line2 = None
        elif intersection.line2 is self:
            intersection.line2 = None
        else: print(f"  couldn't find Line {self.lineNum} in Intersection {intersection.id} to remove it")
        return self

    def reassign_values(self, line_num, num_of_pts, a, b, start, end, points):
        self.lineNum = float(line_num)
        self.numOfPts = int(num_of_pts)
        self.a = np.array(a)
        if np.linalg.norm(end) < np.linalg.norm(start):
            # if line is backwards, flip it and its slope
            self.start, self.end = end, start 
            self.b = -b
        else:
            self.b = np.array(b)
            self.start =np.array(start)
            self.end = np.array(end)
        #self.bSphere = np_cartesian_to_spherical(b) # (r, theta, phi)
        self.set_points(points)
        return self

    def fit_new_points(self, new_pts, new_num=None):
        if new_num == None:
            line_num = self.lineNum
        else:
            line_num = new_num
        values = fit_points(new_pts, line_num)
        self.reassign_values(**values)
        return self

    def split_points(self, point, new_num_1=None, new_num_2=None): # Splits lines points at a given point, new_num_1=before new_num_2=after point
        '''
        Splits lines points at a given point, new_num_1=before new_num_2=after point
        '''
        new_num1, new_num2 = new_num_1, new_num_2
        if new_num_1 == None: new_num1 = self.lineNum+0.1
        if new_num_2 == None: new_num2 = self.lineNum+0.2

        shifted = self.points # lines points
        shifted[['x','y','z']] = shifted[['x','y','z']] - point # center around intersection
        shifted['mag'] = np.linalg.norm(shifted[['x','y','z']], axis=1) # gives point distance from line intersection point
        shifted['proj_self'] = ( shifted[['x','y','z']].to_numpy() @ self.b ) # to determine if point is before or after intersection
        shifted['proj_self'] = shifted['proj_self']/shifted['mag'] # projection of points onto base line
        
        shifted['region'] = np.where(
                shifted['proj_self'] <=0, # if before vertex
                f"{new_num1}", # assign to new line
                f"{new_num2}") # else: leave it in the old line
        self.points = shifted # make sure points are changed in place
        return self.points
    
# **************************************** Mytree class **************************************************************
class Mytree: # lines, vertices, intersections, unassigned
    # region:Mytree
    '''
    lines, vertices, intersections, unassigned
    '''
    def __init__(self, lines:list=None, verts:list=None, intersections:list=None, unassigned:list=None, truth:list=None):
        self.lines = set()
        self.vertices = set()
        self.truth = list()
        self.intersections = set()
        self.unassigned = list() # list of dataframs with unassigned points
        self.num_points = 0
        Vertex._id_counter=1      
        Intersection._id_counter=1  

        if lines is not None:
            for line in lines:
                self.add_line(line)
        if verts is not None:
            for vert in verts:
                self.vertices.add(vert)
                vert.tree = self
        if intersections is not None:
            for inter in intersections:
                self.intersections.add(inter)
        if unassigned is not None:
            for item in unassigned:
                self.unassigned.append(item)
        if truth is not None:
            for item in truth:
                self.truth.append(item)
        return
        

    def add_line(self, line): # Adds Line and its vertices to the tree
        if line in self.lines:
            print(f"  Line {line.lineNum} already in tree")
        else:
            self.num_points += len(line.points)
        self.lines.add(line)
        line.tree = self
        self.sort_lines()
        if line.v1 != None:
            self.vertices.add(line.v1)
        if line.v2 != None:
            self.vertices.add(line.v2) 
        return self
    def add_vetex(self, vertex): # adds vertex and its lines to tree
        if vertex in self.vetices:
            print(f"  Vertex {vertex.id} already in tree")
        self.vertices.add(vertex)
        for line in vertex.lines:
            self.lines.add(line)
        return self
    def add_intersection(self, intersection): # adds intersection and its lines to tree
        if intersection in self.intersections:
            print(f"  Intersection {intersection.id} already in tree")
        self.intersections.add(intersection)
        intersection.tree=self
        self.lines.add(intersection.line1)
        self.lines.add(intersection.line2)
        return self
    def add_truth(self, truth): # adds list of truth points to tree
        for true in truth:
            self.truth.append(true)
    def delete_line(self, Lin):
        if not Lin.points.empty:
            self.unassigned.append(Lin.points)
        Lin.clear()
        self.lines.remove(Lin)
        self.sort_lines()
        return self
    def delete_vertex(self, vertex):
        vertex.clear()
        self.vertices.remove(vertex)
        return self
    def delete_intersection(self, intersection):
        intersection.clear()
        self.intersections.remove(intersection)
        return self
    def clean_lines(self): # removes empty lines 
        new_lines = set()
        for line in self.lines:
            if line.numOfPts == 0:
                line.clear() # make sure line is clear 
                continue
            else:
                new_lines.add(line)
        self.lines = new_lines
        self.sort_lines()
        return new_lines
    def clean_verts(self): # remove empty vertices
        new_verts = set()
        for v in self.vertices:
            if v.lines and v.id != 0: # vert has lines and isn't marked for removal
                new_verts.add(v)
            else: # vert is empty or marked for removal
                v.clear() # make sure actually clear
        self.vertices = new_verts
        return new_verts        
    def clean_intersections(self): # remove empty intersections
        new_inters = set()
        for inter in self.intersections:
            if (inter.line1 and inter.line2) and inter.id != 0: # intersection has lines and isn't marked for removal
                new_inters.add(inter)
            else: # intersection is empty or marked for removal
                inter.clear() # make sure actually clear
        self.intersections = new_inters
        return new_inters  
    def sorted_intersections(self): # sorts intersections based on their score
        return sorted(self.intersections, key=lambda obj:obj.score, reverse=True)
    def sort_lines(self): # sorts intersections based on their score
        pos = 1
        sorted_lines = sorted(self.lines, key=lambda obj:obj.min_z, reverse=False)
        for line in sorted_lines:
            line.position = pos
            pos += 1

        return sorted_lines

    def merge_all_lines(self, refined=False): # attempts to merge every line in the tree
        print(" === Merging lines ===  ")
        for i, base in enumerate(self.lines):
            if base.numOfPts == 0:
                # line is empty, likely already merged
                continue
            for j, line in enumerate(self.lines):
                if j <= i: continue # avoid double checking/merging
                if line.numOfPts == 0:
                    # line is empty, likely already merged
                    continue
                
                if merge_check(base, line, refined=refined): # check if pair of lines should merge
                    merge_lines(base, line) # merges pair of lines
        
        self.clean_lines() # clear out any empty lines
        self.sort_lines()
        return self.lines

    def find_all_intersections(self): # finds the intersections between each line in the tree
        # Intended as the first step in identifying vertices
        print(" === Finding Intersections  ===  ")
        for i, base in enumerate(self.lines):
            #if CONDITION: # skip line for reason
            #    continue
            for j, line in enumerate(self.lines):
                if j <= i: continue # avoid double checking and intersecting with self
                dist = line_infinite_dist(base, line)
                #print(dist, base, line)
                dist_parameter = 25.0 # don't bother if lines are too far apart ---------------------- TODO adjust intersection parameter
                if dist > dist_parameter:
                    continue
                close_pts = closest_points(base, line)
                pt1 = close_pts[0]
                pt2 = close_pts[1]
                inter = Intersection(pt = (pt1+pt2)/2 , line1 = base, point1 = close_pts[0], line2 = line, point2 = close_pts[1])
                self.intersections.add(inter)
                print(f"  {inter} - {base}, {line} - Score: {inter.score}") 
        
        if not self.intersections: print("  Found no intersections  ---------- unexpected")

        return self.intersections
    
    def convert_best_intersection(self):
        print(f" === Converting best Intersection to Vertex === ")
        v = None
        for inter in self.sorted_intersections():
            v = self.intersection_to_vertex(inter)
            if v is not None:
                break
        if v is None: print("  Failed to find valid intersection to convert  ")

    def intersection_to_vertex(self, inter): # converts given intersection into a vertex
        '''
        Splits and re-fits lines that are intersecting
        Converts the intercept to a proper vertex
        Adds new things to tree
        Returns:
            Vertex: converted intersection, was added to tree
        '''
        point = inter.point
        v_rad, new_lines = self.split_intersection(inter)

        v = Vertex(pt=point, rad=v_rad, lines=new_lines, tree=self)
        self.sort_lines()
        return v

    def split_intersection(self, inter):
        '''
        Splits and re-fits lines that are intersecting
        Converts the intercept to a proper vertex
        Adds new things to tree
        Returns:
            Vertex: converted intersection, was added to tree
        '''
        point = inter.point
        on_line1 = point_on_line(point, inter.line1) # check which lines the point is actually on.
        on_line2 = point_on_line(point, inter.line2)

        on_line = [on_line1, on_line2] # for ease of itteration
        lines = [inter.line1, inter.line2]
        other_lines = [inter.line2, inter.line1]

        new_lines = {inter.line1, inter.line2} # lines to be assigned to the resulting vertex
        rads = set()
        new_count = -0.1
        base:Line
        for i, base in enumerate(lines):
            if not on_line[i]: # only split if point is on the line
                continue
            line = other_lines[i]

            if all(on_line):
                rad = 5 # nice safe radius idk ------------------------------------------------- TODO prameter for intersection radius
            else:
                rad=np.linalg.norm(line.start - point) # scorched earth radius ---------------------------------------------------- TODO radius parameter
            rads.add(rad)

            new_count += 0.2
            new_num_1=round(base.lineNum+new_count,2)
            new_num_2=round(base.lineNum+new_count+0.1,2)
            print(f"  {inter} splits {base} into {new_num_1} and {new_num_2}")

            split_lines = self.split_line(point, base, line, rad=rad, new_num_1=new_num_1, new_num_2=new_num_2)
            if len(split_lines) == 0:
                continue
            new_line = split_lines[0]
            new_lines.add(new_line)
            # will make line 1 a true new line, will refit old line with line 2 
        
        # -------------------------------------------------------------------------------------------------------------- TODO impliment vertex conversion better # note from future self, mention whats wrong with it dumbass
        if len(rads) == 0:
            print(f"  {inter} doesn't split its lines, {inter.line1} and {inter.line2}")
            v_rad = 3
            #return None
        else:
            v_rad = sum(rads)/len(rads)
        
        if None in new_lines:
            if len(new_lines)<=2:
                print(f"  lost at least one full line of {inter} ")
                return None
            else:
                print(f" lost a line of {inter} but recoverable")
                new_lines.discard(None)

        return v_rad, new_lines

    def refit_vertices(self):
        if not self.vertices:
            print("  No vertices to refit")
            return

        print(" === Refitting Vertices === ")
        for vert in self.vertices:
            vert.refit()

    def pop_stolen_points(self, point, base, line , rad=None): # returns and adds any points that base 'stole' from line near given point
        if rad == None:
            rad = np.linalg.norm(line.start - point)
        vect = line.start - point
        vect /= np.linalg.norm(vect) # direction of other line
        if 'region' not in base.points.columns:
            split = base.split_points(point)
        else:
            split = base.points
        split['proj_other'] = ( np.dot(split[['x','y','z']].to_numpy(), vect) ) 
        split['proj_other'] = ( split['proj_other']/split['mag'] ) # projection of points onto other line
        
        split['other/self'] = ( split['proj_other']/split['proj_self'] ) # ratio of projections
        split['region'] = np.where( # ------------------------------------------------------------------------------ TODO make a less aggresive version 
                (split['mag'] < rad) & ((split['other/self'] >= 1.0) | (split['other/self'] <= 0.0)), # if within rad and in direction of other line
                'hole', # put in hole
                split['region']) # else leave as is ----------------------------------------------------------------- TODO check double_0 event 110, old func didn't split line 

        split[['x','y','z']] = split[['x','y','z']] + point # shifting everything back
        split.sort_values('z', inplace=True, ignore_index = True) # ensuring sorted well, ------------------------------- TODO maybe sort by mag in the future?
        hole_points = split[split['region'] == 'hole']
        self.unassigned.append(hole_points)
        base.points = split[split['region'] != 'hole']
        return hole_points

    def split_line(self, point, base, line, rad=5.0, new_num_1=None, new_num_2=None): # Splits lines around given intersection
        if new_num_1 == None: new_num1 = base.lineNum+0.1
        else: new_num1 = new_num_1

        if new_num_2 == None: new_num2 = base.lineNum+0.2
        else: new_num2 = new_num_2

        split = base.split_points(point, new_num1, new_num2)
        hole_pts = self.pop_stolen_points(point, base, line, rad=rad)
        
        #print(split)
        #print(new_num1)

        new_pts1 = split[split['region'] == f"{new_num1}"]
        new_pts2 = split[split['region'] == f"{new_num2}"]
        # will make line 1 a true new line, will refit old line with line 2 
        
        #self.unassigned.append(hole_pts) # add points in the hole to set of unassigned points
        #print(new_pts1)
        if new_pts1.empty:
            if hole_pts.empty:
                print(f"  found no change to Line {base.lineNum}. Returning Line {base.lineNum} ---------------------- Unexpected")
                return [base]
            elif new_pts2.empty:
                print(f"  lost all points somehow, Returning Line {base.lineNum} ---------------------- Unexpected")
                return [base]
            elif len(new_pts2) == 1:
                print(f"  found no new line, {new_num1}, and not enough points for {base.lineNum}, unassigning all ---------------------- Unexpected")
                self.delete_line(base)
                return []
            else:
                base.fit_new_points(new_pts2, round(new_num2,2))
                print(f"  found no new line, {new_num1}. Returning {base.lineNum} with hole removed ---------------------- lost a line")
                return [base]
        if len(new_pts1) == 1:
            self.unassigned.append(new_pts1)
            if new_pts2.empty:
                print(f"  lost all but one point, unassigning all ---------------------- lost a line")
                return []
            elif len(new_pts2) == 1:
                print(f" not enough points in either line. Unassigning all ---------------------- lost a line")
                self.unassigned.append(new_pts2)
                base.clear()
                return []
            else:
                print(f" not enough points for line {new_num1}, returning {base.lineNum} with hole removed ---------------------- lost a line")
                base.fit_new_points(new_pts2, round(new_num2,2))
                return [base]
        
        new_line_dict = fit_points(new_pts1, round(new_num1,2)) # fits new points to a line
        new_line = Line(**new_line_dict, tree=self)

        if new_pts2.empty:
            print(f" lost all points in {base}, but new line {new_line} survived ---------------------- lost a line")
            return[new_line]
        elif len(new_pts2) == 1:
            print(f" not enough points left in base. but new line {new_line} survives ---------------------- lost a line")
            self.unassigned.append(new_pts2)
            base.clear()
            return [new_line]
        
        base.fit_new_points(new_pts2, round(new_num2,2))

        return [base, new_line]

    def identify_vertex(self):
        '''merges all lines, finds all intersections, converts best intersection to vertex, refits vertex'''
        self.merge_all_lines()

        self.find_all_intersections()

        self.convert_best_intersection()

        self.refit_vertices()
        
        return

# **************************************** Event class **************************************************************
class Event: 
    '''
    Event object!
    Attributes:
        id : event number
        out_dir : directory to use for saving outputs to
        line_file : file containing hough line data for event
        points_file : file containing point cloud data for event
        points : point cloud dataframe
        tree : custom tree class filled with lines and vertices
        classification : what kind of event it is
    '''
    # region Event
    def __init__(self, event_num:int, out_dir:str, line_file:str = None, points_file:str = None,  points:pd.DataFrame = None, tree:Mytree = None, full_truth=None, classification:str = "unassigned"):
        self.id = event_num
        self.line_file = line_file
        self.points_file = points_file
        self.out_dir = out_dir
        self.data = {}
        self.full_truth = full_truth

        self.classification = classification
        
        if tree is None and line_file is not None:
            self.tree=self.pull_lines_from_file()
        else:
            self.tree = tree
        self.get_truth()

        if points is None:
            if self.points_file is not None:
                self.points=self.pull_points_from_file()
                if self.tree is None:
                    self.tree = Mytree(unassigned=[self.points])
            elif self.tree is not None:
                self.points=self.pull_points_from_tree()
            else:
                print("No points given")
                self.points = None
        else:
            self.points = points
        
        #self.num_hough_lines = len(self.tree.lines)
        return

    def pull_points_from_file(self, in_file=None): # pulls points from point file
        '''Pulls point cloud data from points_file. Assigns to self.points and returns points'''
        if in_file is None:
            file = self.points_file
        else: file=in_file

        if not os.path.isfile(file):
            print(f"  No points file found at {file}, leaving points as they were")
            return None
        points = parse_point_cloud(file)
        self.points = points
        #print(self.points)
        return points
    def pull_lines_from_file(self, in_file=None): # pulls hough lines and puts them in tree
        '''Pulls line data from self.line_file. Creates a Mytree object with lines to assign to self and return'''
        if in_file is None:
            file = self.line_file
        else: file=in_file

        if not os.path.isfile(file):
            print(f"  No line file found at {file}, leaving lines as they were")
            return None
        lines = parse_hough_lines(file)
        tree = Mytree(lines=lines)
        self.tree = tree
        
        
        return tree
    def pull_points_from_tree(self): # pulls points from tree in case no point cloud given
        '''Pulls point cloud data from self.tree. Assigns to self.points and returns points'''
        point_list = []
        for line in self.tree.lines:
            point_list.append(line.points)
        if len(point_list) == 0: return None
        points:pd.DataFrame=pd.concat(point_list)
        points.sort_values('z', ascending=True)
        self.points = points
        return points
    def plot(self, quality=150, out_dir=None, name=None, plot_cloud=False):
        ''' '''
        if out_dir is None: out_dir=self.out_dir
        if name is None: name = f'event_{self.id}_lines3D.png'
        elif not name.lower().endswith(".png"): name = name + ".png"

        figs=[]
        if self.tree is not None:
            #print("plotting tree")
            line_fig = graph_tree(self.tree, self.id)
            file = os.path.join(out_dir, name)
            #print(f"saving fig to '{file}'")
            line_fig.savefig(file, dpi=quality)
            figs.append(line_fig)
        
        if self.points is not None and plot_cloud==True:
            #print("plotting cloud")
            cloud_fig = graph(in_pts=self.points, event=self.id)
            cloud_fig.savefig(os.path.join(out_dir, f'event_{self.id}_points3D.png'), dpi=quality)
            #print(cloud_fig)
            figs.append(cloud_fig)
        return figs
    def get_truth(self, full_truth=None):
        if full_truth is None: full_truth = self.full_truth
        if full_truth is None: return None
        #print(f"================{full_truth.loc[self.id]}================")
        
        try:
            truth=[np.array(b) for (a , b) in full_truth.loc[self.id].dropna().items()]
        except KeyError:
            truth=[np.nan]
            

        self.truth = truth
        self.tree.add_truth(truth)
        return truth
    
    def get_line_num(self):
        self.data.update({'num_lines':len(self.tree.lines)})
        return len(self.tree.lines)

    def prelim_classify(self):
        '''
        Attempt to classify the event with quick checks.
        i.e. check if a reaction is out of bounds, before or after
        '''
        z_min = self.points['z'].min()
        z_max = self.points['z'].max()
        z_diff = (z_max-z_min)/100 # percentage
        i=0
        inc=10
        out={}
        while i+inc<=100:
            z1=z_min + z_diff*i
            i=i+inc
            z2=z_min + z_diff*i
            rms = rms_between_zs(self.points,z1, z2)
            col = f"rms {i-inc}%-{i}%"
            out.update({col:rms})
            #out.append(f"  rms {i-inc}%-{i}%: {rms}")
        self.data.update(out)
        #print(out)
        return

    def identify_vertex(self):
        '''runs tree.identify_vertex(): merges all lines, finds all intersections, converts best intersection to vertex, refits vertex'''
        self.tree.identify_vertex()
        return

    def get_dict(self, line_file=True, vertices=True, line_counts=False):
        base = {'event_id' : self.id}

        if line_file:
            base['in_line_file'] = self.line_file

        tree = self.tree
        if vertices:
            verts = {}
            for vert in tree.vertices:
                verts.update({f'vert_{vert.id}_x' : vert.point[0], f'vert_{vert.id}_y' : vert.point[1], f'vert_{vert.id}_z' : vert.point[2]})
            base.update(verts)

        

        return base

# ++++++++++++++++++++++++++++++++++++++++ Functions used by classes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def rms_between_zs(points, z1, z2):
    zs=min(z1,z2)
    ze=max(z1,z2)
    subset:pd.DataFrame = points[(points['z'] >= zs) & (points['z'] <= ze)]
    if subset.empty:
        print(f"  no points within {zs} and {ze} to find rms")
        return -1
    x_rms = np.sqrt((subset['x']**2).mean())
    y_rms = np.sqrt((subset['y']**2).mean())
    rms = np.sqrt(x_rms**2 + y_rms**2)
    #print(subset, x_rms, y_rms)
    return rms

# ======================================== check if two lines should be merged ============================================================
def merge_check(base, line, refined=False): # Input: two lines | Output: True if should merge
    '''
    Checks if line should be merged into base
    Parameters:
        base: Line object
        line: Line object
    Returns:
        Bool: T/F should they merge
    '''
    slope_dist = np.linalg.norm(base.b - line.b)
    if refined:
        slope_param = 0.03 # ---------------------------------------------------- TODO parameter for refined merging similar lines
    else:
        slope_param = 0.06 # ---------------------------------------------------- TODO parameter for merging similar lines
    if slope_dist >= slope_param:
        # if slopes are very different, then lines shouldn't merge
        print(f"  slope dist between {base.lineNum} and {line.lineNum}:   {slope_dist}") # -______________________ TODO comment print
        return False

    zero_ind = np.abs(base.b).argmax() # compares distance between lines in the most perpendicular plane
    if base.b[zero_ind] == 0:
        print(f"  Line {base.lineNum} slope undefined")
        return False
    if line.b[zero_ind] == 0:
        print(f"  Line {line.lineNum} slope undefined")
        return False

    if ( (line.end[zero_ind]-base.end[zero_ind]) / base.b[zero_ind] ) >= 0 : # select the plane with the most "in the middle" endpoint
        end = base.end
        other = line
    else:
        end = line.end
        other = base

    plane = end[zero_ind]
    t = (plane - other.a[zero_ind]) / other.b[zero_ind]
    other_intercept = np.add(other.a, t * other.b)
    intercept_dist = np.linalg.norm(other_intercept - end)
    print(intercept_dist)
    # base_t = -base.start[zero_ind]/base.b[zero_ind]
    # base_intercept = np.add(base.start, base_t * base.b)
    # line_t = -line.start[zero_ind]/line.b[zero_ind]
    # line_intercept = np.add(line.start, line_t * line.b)
    # intercept_dist = np.linalg.norm(base_intercept - line_intercept)
    if refined:
        intercept_param = 5.0 # ----------------------------------------------- TODO parameter refined for how far apart lines can be. Likely needs to be a function itself
    else:
        intercept_param = 10.0 # ----------------------------------------------- TODO parameter for how far apart lines can be. Likely needs to be a function itself
    if intercept_dist >= intercept_param:
        # lines likely too far apart to be merged. 
        print(f"  intercept dist between {base.lineNum} and {line.lineNum}:       {intercept_dist}") # -______________________ TODO comment print
        return False
    print(f"  Line {line.lineNum} should merge into Base {base.lineNum}")
    return True

# ======================================== Merge two line objects into one ============================================================
def merge_lines(base, line): # Input: two lines | merges them into base Line | Output: base Line
    '''
    Merges line into base. Done in place. Returns base if needed, but base is modified
    Parameters:
        base: Line object to absorb line
        line: Line object to merge into base
    Returns:
        base: merged line
    '''
    print(f"  merging line {line.lineNum} into base {base.lineNum}")
    base.remove_vertices() # making new line, so any old vertices will be outdated
                    
    new_pts = pd.concat([base.points, line.points])
    new_pts.sort_values('z', inplace=True, ignore_index = True)
    base.fit_new_points(new_pts)
    line.clear() # clear line so it can be removed after merging is done
    return base

# ======================================== finds the closest distance between two *infinite* lines ============================================================
def line_infinite_dist(base:Line, line:Line): # Input: base line, reference line | Output: shortest distance between each infinite line
    """
    Finds the shortest distance between given lines. \n Lines treated as infinite, ie extend past start and end 
    Parameters:
        base: first line object
        line: second line object
    Returns:
        dist: distance between lines
    """
    if np.array_equal(base.b, line.b):
        a2 = line.a
        a1 = base.a
        d = base.b
        dist = np.linalg.norm( np.cross((a2-a1), d) ) / np.linalg.norm(d)
    else:
        a2 = line.a
        a1 = base.a
        d2 = line.b
        d1 = base.b
        dist = np.linalg.norm( np.dot( (a2-a1), np.cross(d1, d2) ) ) / np.linalg.norm( np.cross(d1, d2) )
    return dist

# ======================================== finds the points on each *infinite* line closest to the other ============================================================
def closest_points(base:Line, line:Line): # Input: base line, reference line | Output: point on either line closest to the other
    """
    finds the points on each ***infinite**** line closest to the other line
    Parameters:
        base: base Line obj
        line: other Line obj
    Returns:
        list: [point on base, point on line]
    """
    p1 = base.a
    d1 = base.b
    p2 = line.a
    d2 = line.b
    
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b**2
    if np.isclose(denom, 0): # if lines are parallel
        t = 0
        s = d / b if not np.isclose(b, 0) else 0
    else:
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom

    pt1 = p1 + t*d1
    pt2 = p2 + s*d2

    return [pt1, pt2]

# ======================================== determines if given point is 'on' the given line ============================================================
def point_on_line(pt, line): # Input: point, line | Output: True if points on the line
    '''
    Checks if a point is within the start and end of a line.
    Parameters:
        pt: numpy array of a (x, y, z) point
        line: Line object to check if pt is 'on'
    Returns:
        bool: T/F if in line or not
    '''
    # 'on' meaning it is within the start and end of the line.
    shifted_start = pt-line.start # shifts point to be centered around the start of the line
    proj_start = np.dot((shifted_start), line.b) # projects shifted vector along line
    if proj_start < 0: return False # point is before the line

    shifted_end = pt-line.end # shifts point to be centered around the end of the line
    proj_end = np.dot((shifted_end), line.b) # projects shifted vector along line
    if proj_end > 0: return False # point is after the line
    
    return True

# ======================================== fit points to a line ============================================================
def fit_points(base_pts, line_num=0): # Input: df of points, desired line number | Output: dict for Line construction
    '''
    Fits given points to a line and defines values needed for Line object
    Parameters:
        base_pts: df of points to be fit
        line_num: line number for new line, def=0
    Returns:
        dict: {'line_num', 'num_of_pts', 'a', 'b', 'start', 'end', 'points'}
    '''
    
    # outputs dict needed to create a new line. might change
    sorted_pts=base_pts.sort_values('z', ignore_index = True)
    pts = sorted_pts[['x','y','z']].to_numpy()
    centroid = np.mean(pts, axis=0)
    uu, dd, vv = np.linalg.svd(pts - centroid)
    a=centroid
    b = vv[0]
    b /= np.linalg.norm(b)
    z_min=pts[0][2]
    
    #print(f"{z_min}, {a[2]}, {b[2]}")
    t_min=(z_min - a[2])/b[2]
    start = np.array([b[0]*t_min + a[0], b[1]*t_min + a[1], z_min])
    z_max=pts[-1][2]
    t_max=(z_max - a[2])/b[2]
    end = np.array([b[0]*t_max + a[0], b[1]*t_max + a[1], z_max])

    if np.linalg.norm(end) < np.linalg.norm(start):
        # if line is backwards, flip it.
        start, end = end, start
    if (np.dot(b,(end-start)) <= 0): 
        # slope pointing wrong way, flip it
        b = -b

    return {'line_num':line_num, 'num_of_pts' : int(base_pts['z'].count()), 'a' : a, 'b' : b, 'start' : start, 'end' : end, 'points' : base_pts} 


# ++++++++++++++++++++++++++++++++++++++++ Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ======================================== parse line data from --linesdir ============================================================
def parse_hough_lines(path): # Input: lines.txt file | Output: list of Line objects
    """
    Parses line data from lines.txt
    
    Headers per hough line look like:
        Line i: n=j a=(ax,ay,az) b=(bx,by,bz) start=(sx,sy,sz) end=(ex,ey,ez) 
    Data looks like 6 floats seperated by spaces:
        x y z globalTime peakAmplitude integratedCharge
    Parameters:
        path: lines.txt file with line and point data
    Returns:
        lines:list of Line objects
    """

    lines = []
    current = None
    header_re = re.compile(r"#\s*Line\s*(\d+).*n=(\d+)\s*a=\(\s*([^)]+?)\s*\).*b=\(\s*([^)]+?)\s*\).*start=\(\s*([^)]+?)\s*\).*end=\(\s*([^)]+?)\s*\).*", re.IGNORECASE) # \s* matches whitespace characters, \d+ matches digits, .* matches any sequence of charachters until the next token, ([^)]+?) is a capture group: [^)] matches anything except a close parenthesis, the + lets it match a chain of charachters, the ? makes it take the shortest possible string.
    with open(path) as file:
        for lineno, raw in enumerate(file, 1):
            line = raw.strip()
            head = header_re.search(line)
            if head: # if code hit a header in the data, it assigns the apropriate header data to current and adds it to the list of lines
                if current is not None: #if we have found a new header, convert the old list of points to a pd DataFrame
                    current['points']=pd.DataFrame(data=current['points'])
                    curr_line = Line(**current) # converts dict to custom Line object
                    lines.append(curr_line)
                hough_line_num = int(head.group(1))
                num_of_pts = int(head.group(2))
                a = np.fromstring(head.group(3), sep=',', dtype=np.float64)
                b = np.fromstring(head.group(4), sep=',', dtype=np.float64)
                start = np.fromstring(head.group(5), sep=',', dtype=np.float64)
                end = np.fromstring(head.group(6), sep=',', dtype=np.float64)
                direction = np.linalg.norm(end) - np.linalg.norm(start)
                if direction <0:
                    start, end = end, start
                    b=-b
                current = {"line_num": hough_line_num, 'num_of_pts' : num_of_pts,  'a' : a, 'b' : b, 'start' : start, 'end' : end, 'points' : {'x':[], 'y':[], 'z':[], 'gTime' : [], 'peakAmp' : [], 'integQ' : []}}
                #lines.append(current)
                continue
            if line.startswith('#') or not line: # if the code hits a line that is header without line info, or if it cant find a line, it skips the line
                continue
            data = line.split()
            if len(data) ==6 and current is not None: # makes sure the number of data points is as expected, and ensures that there is a hough line currently being "looked at"
                try:
                    i = 0
                    for value in current['points'].values(): # rather than manually appending each list in 'points' this itterates through and places each data point in its apropriate list
                        value.append(float(data[i]))
                        i+=1
                except ValueError:
                    print(f"	[parse warning] line {lineno} not numeric: {line!r}")
            else:
                print(f"	[parse warning] skipping line {lineno}: {line!r}")
                continue
    if current is not None:
        current['points']=pd.DataFrame(data=current['points']) # converts the last lines points to a pd DF
        curr_line = Line(**current)
        lines.append(curr_line)

    print(f"	Parsed {len(lines)} lines from {os.path.basename(path)}")
    #print(lines)
    return lines

def parse_point_cloud(path):
    '''
    Parses point cloud data from event_#.dat file
    Parameters:
        path: event_#.dat file with point data
    Returns:
        points_df: pandas dataframe of all points
    '''
    points = {'x':[], 'y':[], 'z':[], 'gTime' : [], 'peakAmp' : [], 'integQ' : []}
    with open(path) as file:
        for i, raw in enumerate(file):
            if i == 0: 
                if "x_fit" not in raw:
                    print("  point header seems incorrect, data may be lost or wrong")
                continue # skip first line. Should be header info
            line = raw.strip()
            data = line.split()
            if len(data) < 6: 
                print(f"  bad data line in {file}, line number {i}, skipping line")
                continue
            for j, value in enumerate(points.values()):
                value.append(float(data[j]))
    points_df = pd.DataFrame(data=points)
    return points_df

# ======================================== removes out of bounds points below certain y value ============================================================ TODO Update
def nuke(line:Line): 
    #print(len(line.points.index))
    line.points.drop(line.points[line.points['y'] <= -70].index, inplace=True)
    temp = line.numOfPts
    line.numOfPts = len(line.points.index)
    if temp != line.numOfPts:
        print(f"  {line} had {temp-line.numOfPts} points removed for being outside of bounds due to reconstruction issues")
    #print(len(line.points.index))
    return

# ======================================== Finds the closest point to a reference and the distance  ============================================================
def closer(ref_pt, pts:list): # Input: ref point, list of points [point 1, point 2, ...] | Output: [closer_pt, dist]
    """
    Finds the closest point to a reference and the distance
    Parameters:
        ref_pt: np.array reference point
        pts: list of points to check
    Returns:
        list: [closer_point, distance]
    """
    dist=None
    for pt in pts:
        shift = pt-ref_pt
        new_dist = np.linalg.norm(shift)
        if dist == None:
            dist = new_dist
            closer_pt = pt
        elif new_dist < dist:
            dist = new_dist
            closer_pt = pt
    return [closer_pt, dist]


# ++++++++++++++++++++++++++++++++++++++++ graphing funcs +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# region: Graphing

# ======================================== line graphing func ============================================================    
def graph(in_lines = [], in_circles = [], in_pts = [], in_xs = [], in_tris = [], event=-1): # Input: *list of Line objects*, *list of np.array points*, *current event number* | Output: fig object 
    # plots given lines and points, titles with the current event 
    
    lines = in_lines
    pts = in_pts
    circles = in_circles
    xs = in_xs
    tris = in_tris
    if type(pts) == pd.DataFrame:
        pts = pts[['x', 'y', 'z']].to_numpy()

    plt.clf()
    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(17,9),num=event,constrained_layout=True)
    plot_angles = [[90,-90],[0,0],[45,-45]]
    rowcolind = 131
    for angle in plot_angles:
        ax = fig.add_subplot(rowcolind, projection='3d')
        ax.view_init(elev=angle[0], azim=angle[1])
        rowcolind = rowcolind + 1

        for i, line in enumerate(lines):
            #print(line.points['x'])
            color = cmap(i % cmap.N)
            start, end = line.start, line.end
            ax.scatter(line.points['x'], line.points['z'], line.points['y'], color=color, edgecolors='black', alpha = 1, s=7, linewidths=0.5)
            ax.plot([start[0], end[0]], [start[2], end[2]], [start[1], end[1]], color=color, linewidth=2, label=f"Line {line.lineNum}") #z and y are flipped bc z is beam axis
            ax.legend(loc='best')
        for pt in pts:
            ax.scatter(pt[0], pt[2], pt[1], facecolors='black', edgecolors='black', s=3)
        for pt in tris:
            ax.scatter(pt[0], pt[2], pt[1], facecolors='none', edgecolors='red', s=30, marker='^')
        for pt in circles:
            ax.scatter(pt[0], pt[2], pt[1], facecolors='none', edgecolors='blue', s=40)
        for pt in xs:
            ax.scatter(pt[0], pt[2], pt[1], marker='x', color='black')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_zlabel('Y (mm)')
        ax.set_xlim(-150, 150)
        ax.set_ylim(45, 350)
        ax.set_zlim(-30, 70)
        ax.set_title(f'Event {event}')
        
    return fig

# ======================================== tree graphing func ============================================================   TODO add better option selection
def graph_tree(tree, event=-1): # Input: tree with lines, verts, and intersects to graph, *event number* | Output: figure object
    # graphs all lines and vertices in tree. might add intersections later
    graph_lines = tree.lines
    points = []
    circles = []
    xs = []
    tris = []
    for v in tree.vertices:
        circles.append(v.point)
    for true in tree.truth:
        tris.append(true)
    if len(tree.unassigned) != 0:
        free_pts_df = pd.concat(tree.unassigned)
        points = free_pts_df[['x', 'y', 'z']].to_numpy()
    for i in tree.intersections:
        xs.append(i.point)
    line_fig = graph(in_lines=graph_lines, in_circles=circles, in_pts=points, in_xs=xs, in_tris=tris, event=event)
    return line_fig
# endregion


# ++++++++++++++++++++++++++++++++++++++++ Vertex Assesment +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# region: Vertex Assesment

# ======================================== pulls all the vertices for event from full truth DF ============================================================
def my_truth(full_truth, event): # Input: full_truth DataFrame, event # | Output: list of dicts {'id', 'point'}
    true_points = full_truth.loc[event].tolist()
    true_verts = []
    for i, pt in enumerate(true_points):
        true_vert = {'id' : i, 'point' : pt}
        true_verts.append(true_vert)
    return true_verts

# ======================================== pulls all the true vertices from ROOT file ============================================================
def get_truth(path, only_siHitE=False): # Input: path to truth ROOT file | Output: DataFrame of 'reaction' and 'decay' vertices (index is event #)
    # someday should be more general 
    if path is None: return None
    file = ROOT.TFile.Open(path) # open root file
    root_tree = file.Get("simData") # tree should be named simData
    root_tree.SetBranchStatus("*", 0)
    root_tree.SetBranchStatus("siHitE", 1)
    root_tree.SetBranchStatus("vertexX", 1)
    root_tree.SetBranchStatus("vertexY", 1)
    root_tree.SetBranchStatus("vertexZ", 1)
    root_tree.SetBranchStatus("decayVtxX_mm", 1)
    root_tree.SetBranchStatus("decayVtxY_mm", 1)
    root_tree.SetBranchStatus("decayVtxZ_mm", 1)

    reaction_pts = [] # list of reaction vertices
    decay_pts = [] # list of decay vertices
    for entry in root_tree: # each entry is an event
        if only_siHitE and len(entry.siHitE) == 0: # if there was no Silicon Hit Energy, ignore the event
            reaction_pts.append(np.nan) # adding nan so the index can be used for the event number
            decay_pts.append(np.nan)
            continue
        vx1 = entry.vertexX # vertex_ is the reaction vertex
        vy1 = entry.vertexY
        vz1 = entry.vertexZ
        reaction_pts.append(np.array((vx1,vy1,vz1)))
        if len(entry.decayVtxX_mm) != 1: # if the vertex data is messed up or missing, toss
            decay_pts.append(np.nan)
        else:
            vx2 = entry.decayVtxX_mm[0] # decayVtx_ is the decay vertex
            vy2 = entry.decayVtxY_mm[0]
            vz2 = entry.decayVtxZ_mm[0]
            decay_pts.append(np.array((vx2,vy2,vz2)))
    points = pd.DataFrame({'reaction':reaction_pts, 'decay':decay_pts})
    points = points.dropna(how='all') # removes rows that are 'all' nan, keeps partial rows
    #print(points)
    return points

def add_to_truth(truth, verts): # add vertices to truth, simple for single vert for now TODO

    return truth

def get_truth_distances(truth, far_param):
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)

    num_events = len(truth.index)

    missed = truth[truth['found'].isna()]
    #print(truth)
    print(f"\n  === missed events: === \n{missed}")

    truth.dropna(subset=['found'], inplace=True)
    truth['true-mine']=truth['reaction']-truth['found']
    truth['dist']=truth['true-mine'].apply(np.linalg.norm)

    true_arr = np.vstack(truth['reaction'])
    found_arr = np.vstack(truth['found'])
    diff =  found_arr - true_arr 
    truth[['dx', 'dy', 'dz']] = diff
    num_far = (truth['dist'] > far_param).sum()
    
    #print(truth)
    print(f" number of events {num_events}")
    print(f" number missed {len(missed)}")
    print(f" number farther than {far_param}mm {num_far}")
    truth.drop(truth[truth['dist'] > far_param].index, inplace=True)
    return truth

# endregion

# ======================================== Cartesian to sphereical converter helper function ============================================================
def np_cartesian_to_spherical(point): # Input: np.array(x,y,z) | Output: np.array(r,theta,phi)
    x=point[0]
    y=point[1]
    z=point[2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arccos(np.where(r == 0, 0, z/r)))
    phi = np.degrees(np.arctan2(y, x))
    return np.array([r,theta,phi])


# ++++++++++++++++++++++++++++++++++++++++ File/argument managment +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def in_dir(path):
    if not os.path.isdir(path):
        sys.exit(f"  Couldn't file directory: {path}")
    return path

def out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def in_file(path):
    if not os.path.isfile(path):
        sys.exit(f"  Couldn't find file: {path}")
    return path

def out_file(path):
    if os.path.isfile(path): return path
    try:
        with open(path, 'w'): # attempt to create the file
            return path
    except FileNotFoundError:
        sys.exit(f"  Couldn't create file at: {path} \n    Likely bad path")
    return path


