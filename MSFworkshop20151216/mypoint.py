class mypoint(object):
    def __init__(self,x=0.,y=0.,z=0.):
        self.x = x
        self.y = y
        self.z = z
    def show(self):
        print("x=%f"%self.x)
    def getx(self):
        return self.x
    def setx(self,newx):
        self.x=newx
    def __repr__(self):
        return "x=%f y=%f z=%f"%(self.x,self.y,self.z)

print "loaded!"
