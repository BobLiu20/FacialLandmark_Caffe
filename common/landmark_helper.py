# coding=utf-8
'''
Bob.Liu in 20171114
'''
import numpy as np
import cv2

class LandmarkHelper(object):
    '''
    Helper for different landmark type
    '''
    @classmethod
    def parse(cls, line, landmark_type):
        '''
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input txt
            landmark_type: len of landmarks
        Return:
            see child parse
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            return cls.__landmark5_txt_parse(line)
        elif landmark_type == 83:
            return cls.__landmark83_txt_parse(line)
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def flip(a, landmark_type):
        '''
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: len of landmarks
        Returns:
            landmarks: new landmarks
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            landmarks = np.concatenate((a[1,:], a[0,:], a[2,:], a[4,:], a[3,:]), axis=0)
        elif landmark_type == 83:
            landmarks = np.concatenate((a[10:19][::-1], a[9:10], a[0:9][::-1], a[35:36],
                a[36:43][::-1], a[43:48][::-1], a[48:51][::-1], a[19:20], a[20:27][::-1],
                a[27:32][::-1], a[32:35][::-1], a[56:60][::-1], a[55:56], a[51:55][::-1],
                a[60:61], a[61:72][::-1], a[72:73], a[73:78][::-1], a[80:81], a[81:82],
                a[78:79], a[79:80], a[82:83]), axis=0)
        else:
            raise Exception("Unsupport landmark type...")
        return landmarks.reshape([-1, 2])

    @staticmethod
    def __landmark5_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=[0:4] is bbox and [4:] is landmarks
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        data = map(int, a[1:])
        pts = data[4:] # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark83_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks83, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        a1 = np.fromstring(a[1], dtype=int, count=166, sep=',')
        a1 = a1.reshape((-1, 2))
        return a[0], a1

