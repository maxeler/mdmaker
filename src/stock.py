#!/usr/bin/python

import sys, getopt
import numpy
import random
from matplotlib import pyplot
from array import *
from collections import deque
import warnings

# all new resting orders will be placed qith qty 1
# book is kept with at least one level on bid and offer

minSpread = 1
decimals = 1
minDepth = 5
orderQ = 100

# check that two floats are within a margin of error
error_margin = 0.01
def match(p1, p2):
	return abs(p1 - p2) < error_margin

class Level:
	def __init__(self, px, qty):
		self.price = px
		self.quantity = qty

class Side:
	def __init__(self, incrementing, debug):
		self.levels = deque()
		self.incrementing = incrementing
		self.name = "BID"
                if self.incrementing: self.name = "OFFER"
		self.operation = "SELL"
                if self.incrementing: self.operation = "BUY"
		self.debug = debug

	def execute(self, price):
		if len(self.levels) > 0:
			if self.incrementing: # OFFER
				if price >= self.levels[0].price - error_margin:
					if self.debug:
						print("%s for %f at level %f" % (self.operation, price, self.levels[0].price))
					self.levels.popleft()
					return True
				return False
			else:
				if price <= self.levels[0].price + error_margin:
					if self.debug:
						print("%s for %f at level %f" % (self.operation, price, self.levels[0].price))
					self.levels.popleft()
					return True
				return False
		else:
			return False

	def isBetter(self, price):
		if self.incrementing:
			return price > self.levels[0].price
		else:
			return price < self.levels[0].price

	def add(self, price):
		price = round(price, decimals)
		if self.debug: print("Add to %s: %f" % (self.name, price))
		if len(self.levels) == 0:
			self.levels.append(Level(price, orderQ))
			return
		outQue = deque()
		added = False
		for level in self.levels:
			if added == False:
				if match(price, level.price):
					outQue.append(Level(level.price, level.quantity + orderQ))
					added = True
				elif self.incrementing: # OFFER
					if price < level.price:
						outQue.append(Level(price, orderQ))
						added = True
					outQue.append(level)
				else:
					if price > level.price:
						outQue.append(Level(price, orderQ))
						added = True
					outQue.append(level)
			else:
				outQue.append(level)
		if added == False:
			outQue.append(Level(price, orderQ))
		self.levels = outQue

	def depth(self):
		return len(self.levels)
	
	def top(self):
		return self.levels[0]

	def printStr(self, index):
		if index < self.depth():
			if self.incrementing:
				return "%f,%u,1" % (self.levels[index].price, self.levels[index].quantity)
			else:
				return "1,%u,%f" % (self.levels[index].quantity, self.levels[index].price)
		else:
			return "0,0,0"

class Book:
	def __init__(self, filename, security, debug):
		self.bid = Side(False, debug)
		self.offer = Side(True, debug)
		self.md = open(filename, 'w')
		self.md.write("Security,bid valid,bid qty,bid px,ask px,ask qty,ask valid,Timestamp,execSummary,Price,DisplayQuantity,AggressorSide(1-buy,2-sell)\n")
		self.security = security
		self.timestamp = 100000000L
		self.debug = debug

	def incTime(self, bigStep):
		if bigStep:
			self.timestamp = self.timestamp + 100000000L
		else:
			self.timestamp = self.timestamp + 1000000L

	def buy(self, price, doUpdate=True):
		self.incTime(False)
		qty = 0
		if self.offer.depth() > 0:
			qty = self.offer.levels[0].quantity;	

		if self.offer.execute(price) == False:
			self.bid.add(price)
			if doUpdate:
				self.writeTopOfBook(0, 0, 0, 0)
		else:
			self.writeTopOfBook(1, price, qty, 1)

	def sell(self, price, doUpdate=True):
		self.incTime(False)
		qty = 0
		if self.bid.depth() > 0:
			qty = self.bid.levels[0].quantity;	

		if self.bid.execute(price) == False:
			self.offer.add(price)
			if doUpdate:
				self.writeTopOfBook(0, 0, 0, 0)
		else:
			self.writeTopOfBook(1, price, qty, 2)

	def addToChange(self, midDiff):
		if midDiff > 0:
			self.buy(self.bid.top().price + midDiff)
		elif midDiff < 0:
			self.sell(self.offer.top().price + midDiff)

	def executeToChange(self, midDiff):
		if midDiff > 0:
			top = self.offer.top().price
			if self.offer.depth() < minDepth:
				self.sell(top + 2 * midDiff, False)
			self.buy(top + midDiff)
		elif midDiff < 0:
			top = self.bid.top().price
			if self.bid.depth() < minDepth:
				self.buy(top + 2 * midDiff, False)
			self.sell(top + midDiff)

	def moveMid(self, targetMid):
		if self.debug:
			print "moving mid from %f to %f" % (book.mid(), targetMid)
		loops = 0
		while match(targetMid, self.mid()) == False:
			mid = self.mid()
			spread = self.spread()
			if self.debug:
				print "current mid %f target mid %f spread%f" % (mid, targetMid, spread)
			midDiff = targetMid - mid
			if midDiff < spread / 1.5 and spread > minSpread:
				self.addToChange(2 * midDiff)
			else:
				self.executeToChange(midDiff)
			loops = loops + 1
			if loops > 10:
				warnings.warn("Could not reach target mid")
				break
		if self.debug:
			print "mid is now %f" % book.mid()
			self.prints()
			print "---"

	def mid(self):
		if self.bid.depth() > 0 and self.offer.depth() > 0:
			return (self.bid.top().price + self.offer.top().price) / 2
		else:
			raise IndexError("Single sided book (mid)")

	def spread(self):
		if self.bid.depth() > 0 and self.offer.depth() > 0:
			return self.offer.top().price - self.bid.top().price
		else:
			raise IndexError("Single sided book (spread)")

	def writeTopOfBook(self, isExec, price, quantity, side):
		# security, bid valid, bid qty, bid px, offerpx, offer qty, offer valid, timestamp, execSummary, price, qty, side
		self.md.write("%d,%s,%s,%d,%d,%f,%u,%d\n" % (self.security, self.bid.printStr(0), self.offer.printStr(0), self.timestamp, isExec, price, quantity, side))

	def prints(self):
		maxLen = self.bid.depth()
		if self.offer.depth() > maxLen:
			maxLen = self.offer.depth()
		for i in range(0, maxLen):
			print "[%s | %s]" % (self.bid.printStr(i), self.offer.printStr(i))

def main(argv):
	print "Market Data Generator"
	
	try:
		opts, args = getopt.getopt(argv, "s:i:v:b:e:f:d", ["samples=", "base=", "instrument=", "variation=", "--seed", "--file"])
	except getopt.GetoptError:
		print "-s --samples= <num> -b --base <starting price> -i --instrument= <security id> -v --variation= <change in price> -d <debug> -e --seed <seed> -f --file <file>"
		sys.exit(2)
	
	samples = 10000L
	base = 200.0
	variation = 1.0
	instrument = 123456L
	debug = False
	seed = 1
	filename = "md.csv"
	
	for opt, arg in opts:
		if opt == "-d":
			debug = True
		if opt in ("-s", "--samples"):
			samples = long(arg)
		elif opt in ("-b", "--base"):
			base = float(arg)
		elif opt in ("-i", "--instrument"):
			instrument = long(arg)
		elif opt in ("-v", "--variation"):
			variation = float(arg)
		elif opt in ("-e", "--seed"):
			seed = arg
		elif opt in ("-f", "--file"):
			filename = arg
	
	random.seed(6)
	direction = 1
	idealMid = [ ] 
	bookMid = [ ] 
	pos = base
	bound = 100.0
	
	#book = Book("stxe.csv", 1548173, False)
	book = Book(filename, instrument, debug)
	random.seed(seed)
	
	for i in range(samples):
		if (i % 10 == 0): 
			if pos > (base + bound): 
				direction = -1.0;
			elif pos < (base - bound):
				direction = 1.0
			elif random.randint(0, 1) == 0: 
				direction = -1.0
			else:
				direction = 1.0
		pos += direction * (random.uniform(0, variation) / 10.0)
		if pos <= 0:
			pos = 0
		pos = round(pos, decimals)
		if len(idealMid) > 0:
			try:
				book.moveMid(pos)
			except IndexError:
				print "Book after IndexError raised"
				book.prints()
				raise
		else:
			midDiff = 0
			book.buy(pos - minSpread) # first entry
			book.sell(pos + minSpread)
			book.buy(pos - 2 * minSpread) # first entry
			book.sell(pos + 2 * minSpread)
			book.prints()
	
		bookMid += [book.mid()]
		idealMid += [pos]
		book.incTime(True)
	
	print("Final book")
	book.prints()
		
	#h = numpy.array(idealMid)
	#print h.shape
	pyplot.plot(range(samples), numpy.array(idealMid), 'b', range(samples), numpy.array(bookMid), 'r--')
	pyplot.show()
	
if __name__ == "__main__":
	main(sys.argv[1:])
