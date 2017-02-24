#!/usr/bin/python3
"""
Market data generator script.
"""

import sys
import random
import warnings
import argparse
from collections import namedtuple
from collections import deque

import numpy
from matplotlib import pyplot


# all new resting orders will be placed qith qty 1
# book is kept with at least one level on bid and offer

MIN_SPREAD = 1
DECIMALS = 1
MIN_DEPTH = 5
ORDER_QUANTITY = 1
ERROR_MARGIN = 0.01


def match(price1, price2):
    """Checks if prices are the same."""
    return abs(price1 - price2) < ERROR_MARGIN


Level = namedtuple('Level', ['price', 'quantity'])


class Side(object):

    def __init__(self, incrementing, debug):
        self.levels = deque()
        self.incrementing = incrementing
        self.name = "BID"
        if self.incrementing:
            self.name = "OFFER"
        self.operation = "SELL"
        if self.incrementing:
            self.operation = "BUY"
        self.debug = debug

    def execute(self, price):
        if len(self.levels) > 0:
            if self.incrementing:  # OFFER
                if price >= self.levels[0].price - ERROR_MARGIN:
                    if self.debug:
                        print("%s for %f at level %f" %
                              (self.operation, price, self.levels[0].price))
                    self.levels.popleft()
                    return True
                return False
            else:
                if price <= self.levels[0].price + ERROR_MARGIN:
                    if self.debug:
                        print("%s for %f at level %f" %
                              (self.operation, price, self.levels[0].price))
                    self.levels.popleft()
                    return True
                return False
        else:
            return False

    def better(self, price):
        if self.incrementing:
            return price > self.levels[0].price
        else:
            return price < self.levels[0].price

    def add(self, price):
        price = round(price, DECIMALS)
        if self.debug:
            print("Add to %s: %f" % (self.name, price))

        if len(self.levels) == 0:
            self.levels.append(Level(price, ORDER_QUANTITY))
            return
        out_queue = deque()

        added = False
        for level in self.levels:
            if added is False:
                if match(price, level.price):
                    out_queue.append(
                        Level(level.price, level.quantity + ORDER_QUANTITY))
                    added = True
                elif self.incrementing:  # OFFER
                    if price < level.price:
                        out_queue.append(Level(price, ORDER_QUANTITY))
                        added = True
                    out_queue.append(level)
                else:
                    if price > level.price:
                        out_queue.append(Level(price, ORDER_QUANTITY))
                        added = True
                    out_queue.append(level)
            else:
                out_queue.append(level)
        if added is False:
            out_queue.append(Level(price, ORDER_QUANTITY))
        self.levels = out_queue

    def depth(self):
        return len(self.levels)

    def top(self):
        return self.levels[0]

    def to_str(self, index):
        if index < self.depth():
            if self.incrementing:
                return "%f,%u,1" % (self.levels[index].price, self.levels[index].quantity)
            else:
                return "1,%u,%f" % (self.levels[index].quantity, self.levels[index].price)
        else:
            return "0,0,0"


class Book(object):

    def __init__(self, outputfile, security, debug):
        self.bid = Side(False, debug)
        self.offer = Side(True, debug)
        self.file = outputfile
        self.file.write(
            "Security, bid valid, bid qty, bid px, "
            "ask px, ask qty, ask valid, "
            "Timestamp, execSummary, Price, "
            "DisplayQuantity, AggressorSide(1-buy,2-sell)\n")
        self.security = security
        self.timestamp = 100000000
        self.debug = debug
        self.timeseries = []

    def incTime(self, big_step):
        if big_step:
            self.timestamp = self.timestamp + 100000000
        else:
            self.timestamp = self.timestamp + 1000000

    def buy(self, price, doUpdate=True):
        self.incTime(False)
        qty = 0
        if self.offer.depth() > 0:
            qty = self.offer.levels[0].quantity

        if not self.offer.execute(price):
            self.bid.add(price)
            if doUpdate:
                self.write_top(0, 0, 0, 0)
        else:
            self.write_top(1, price, qty, 1)

    def sell(self, price, doUpdate=True):
        self.incTime(False)
        qty = 0
        if self.bid.depth() > 0:
            qty = self.bid.levels[0].quantity

        if not self.bid.execute(price):
            self.offer.add(price)
            if doUpdate:
                self.write_top(0, 0, 0, 0)
        else:
            self.write_top(1, price, qty, 2)

    def addToChange(self, midDiff):
        if midDiff > 0:
            self.buy(self.bid.top().price + midDiff)
        elif midDiff < 0:
            self.sell(self.offer.top().price + midDiff)

    def executeToChange(self, midDiff):
        if midDiff > 0:
            top = self.offer.top().price
            if self.offer.depth() < MIN_DEPTH:
                self.sell(top + 2 * midDiff, False)
            self.buy(top + midDiff)
        elif midDiff < 0:
            top = self.bid.top().price
            if self.bid.depth() < MIN_DEPTH:
                self.buy(top + 2 * midDiff, False)
            self.sell(top + midDiff)

    def move_mid(self, target):
        """Move mid price"""
        if self.debug:
            print("moving mid from %f to %f" % (self.mid(), target))
        loops = 0
        while match(target, self.mid()) is False:
            mid = self.mid()
            spread = self.spread()
            if self.debug:
                print("current mid %f target mid %f spread%f" %
                      (mid, target, spread))
            diff = target - mid
            if diff < spread / 1.5 and spread > MIN_SPREAD:
                self.addToChange(2 * diff)
            else:
                self.executeToChange(diff)
            loops = loops + 1
            if loops > 10:
                warnings.warn("Could not reach target mid")
                break
        self.timeseries += [self.mid()]

        if self.debug:
            print("mid is now %f" % self.mid())
            self.prints()
            print("---")

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

    def write_top(self, isExec, price, quantity, side):
        """Writes the Top of Book to a file"""
        # security, bid valid, bid qty, bid px, offerpx, offer qty, offer
        # valid, timestamp, execSummary, price, qty, side
        self.file.write("%d,%s,%s,%d,%d,%f,%u,%d\n" %
                        (self.security, self.bid.to_str(0), self.offer.to_str(0),
                         self.timestamp, isExec, price, quantity, side))

    def prints(self):
        max_len = self.bid.depth()
        if self.offer.depth() > max_len:
            max_len = self.offer.depth()
        for i in range(0, max_len):
            print("[%s | %s]" % (self.bid.to_str(i), self.offer.to_str(i)))
    def get_series(self):
        return self.timeseries

def gen_mid(config):
    """Generator function for the mid price"""
    upper_bound = config.base + config.bound
    lower_bound = config.base - config.bound
    mid = config.base
    direction = 1.0

    for i in range(config.samples):
        if i % 10 == 0:
            if mid >= upper_bound:
                direction = -1.0
            elif mid <= lower_bound:
                direction = 1.0
            elif random.randint(0, 1) == 0:
                direction = -1.0
            else:
                direction = 1.0

        coefficient = random.uniform(0, config.variation) / (config.variation * 10.0)
        mid += direction * coefficient

        if mid <= lower_bound:
            mid = lower_bound

        if mid >= upper_bound:
            mid = upper_bound

        mid = round(mid, DECIMALS)
        yield mid

def gen_book(config):
    """Generate order book"""
    random.seed(config.randseed)
    ideal_mid = []
    book_mid = []

    book = Book(config.outputfile, config.instrument, config.debug)

    # Initialize book
    mid = config.base
    for i in range(config.depth):
        book.buy(mid - (i+1) * MIN_SPREAD)
        book.sell(mid + (i+1) * MIN_SPREAD)

    for mid in gen_mid(config):
        book_mid += [book.mid()]
        ideal_mid += [mid]
        book.move_mid(mid)
        book.incTime(True)

    return book

def parse_args(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Number of samples required",
                        action="store", type=int, default=10000)

    parser.add_argument("-b", "--base", dest="base",
                        help="Starting price",
                        action="store", type=float, default=200.0)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Book depth",
                        action="store", type=int, default=5)

    parser.add_argument("-i", "--instrument", dest="instrument",
                        help="Security ID",
                        action="store", type=int, default=123456)

    parser.add_argument("-v", "--variation", dest="variation",
                        help="Change in price",
                        action="store", type=float, default=1.0)

    parser.add_argument("-u", "--bound", dest="bound",
                        help="Price bound",
                        action="store", type=float, default=100.0)

    parser.add_argument("-g", "--debug", dest="debug",
                        help="Enable debugging",
                        action="store_true", default=False)

    parser.add_argument("-r", "--randomseed", dest="randseed",
                        help="Random seed",
                        action="store", type=int, default=1)

    parser.add_argument("-o", "--output", dest="outputfile",
                        help="Output file name",
                        action="store", type=argparse.FileType(mode='w'), default="md.csv")

    return parser.parse_args(argv)

def main(argv):
    """ Market data generator entry point """
    print("Market Data Generator")

    args = parse_args(argv)

    book = gen_book(args)

    print("Final book")
    book.prints()

    pyplot.plot(range(args.samples), numpy.array(book.get_series()), 'b')
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv[1:])
