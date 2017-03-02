#!/usr/bin/python3
"""
Market data generator script.
"""

import argparse
import bisect
import csv
import random
import sys
import time
import warnings
from enum import Enum
from functools import reduce

import numpy
from matplotlib import pyplot
from recordclass import recordclass
from sortedcontainers import SortedListWithKey

# all new resting orders will be placed qith qty 1
# book is kept with at least one level on bid and offer

DECIMALS = 1
ORDER_QUANTITY = 1
ERROR_MARGIN = 0.01


def match(price1, price2):
    """Checks if prices are the same."""
    return abs(price1 - price2) < ERROR_MARGIN

def now_nanos():
    """Returns the current time since epoch in nanoseconds"""
    return int(time.time() * 1000000000)

class Side(Enum):
    """
        Trading Side
    """
    BUY = 1
    SELL = 2

OrderBookLevel = recordclass('OrderBookLevel',
                             ['price', 'qty', 'order_count'])
Order = recordclass('Order', ['secid', 'side', 'price', 'qty'])
Trade = recordclass('Trade', ['price', 'qty', 'aggressor'])

class OrderBook(object):
    """An Order Book data model"""
    def __init__(self, security):
        self.bid = SortedListWithKey(key=(lambda level: level.price))
        self.offer = SortedListWithKey(key=(lambda level: level.price))
        self.security = security

    def cross(self, bid, offer):
        """
            Check if two levels can cross
        """
        if bid.price >= offer.price:
            remaining_qty = offer.qty - bid.qty
            return (True, remaining_qty)

        return (False, 0)

    def match(self, aggressor_side):
        """
            Match orders and return a list of trades
        """

        # print("Matching on the following book:")
        # self.print()
        trades = []
        for bid_i in range(len(self.bid) - 1, -1, -1):
            bid = self.bid[bid_i]
            size_offer = len(self.offer)
            offer_i = 0
            while offer_i < size_offer:
                offer = self.offer[offer_i]
                (crossed, remaining_qty) = self.cross(bid, offer)
                if crossed:
                    trade = Trade(price=offer.price, qty=offer.qty, aggressor=aggressor_side)
                    stop = False
                    if remaining_qty >= 0:
                        offer.qty = remaining_qty
                        trade.qty = bid.qty
                        del self.bid[bid_i]
                        stop = True
                    if remaining_qty <= 0:
                        bid.qty = abs(remaining_qty)
                        del self.offer[offer_i]
                        size_offer -= 1
                    else:
                        offer_i += 1
                    trades += [trade]
                    if stop:
                        break
                else:
                    return trades
        return trades


    def compact(self, levels):
        """
            Compacts an order list, such that orders at the same price level are merged.
            This assumes the order list is sorted.
        """

        # print("Compacting book")
        # self.print()
        last_level = None
        for i in range(len(levels) - 1, -1, -1):
            level = levels[i]
            if last_level:
                if level.price == last_level.price:
                    last_level.qty += level.qty
                    last_level.order_count += level.order_count
                    del levels[i]
                else:
                    last_level = level
            else:
                last_level = level

    def order(self, order):
        """
            Apply an order to the Order Book.
            Return a list of trades generated as a result.
        """

        print("Evaluating order: ", order)
        if self.security != order.secid:
            raise ("Cannot place order for security "
                   "%s on book[%s]" % (order.security, self.security))

        levels = self.bid
        if order.side == Side.SELL:
            levels = self.offer

        levels.add(OrderBookLevel(price=order.price, qty=order.qty, order_count=1))
        self.compact(levels)
        return self.match(order.side)

    def mid(self):
        """
            Get mid price of top of Book
        """
        if self.bid and self.offer:
            return (self.bid[-1].price + self.offer[0].price) / 2.0

        return 0

    def reduce_order_count(self, level1, level2):
        """
            Reduction function for order_counts
        """
        count1 = level1
        count2 = level2
        if isinstance(level1, OrderBookLevel):
            count1 = level1.order_count
        if isinstance(level2, OrderBookLevel):
            count2 = level2.order_count
        return count1 + count2

    def num_orders(self, levels):
        """
            Computes the number of open orders on a list of book levels
        """
        return reduce(self.reduce_order_count, levels)

    def num_offers(self):
        """
            Returns the number of offers on the book
        """
        return self.num_orders(self.offer)

    def num_bids(self):
        """
            Returns the number of bids on the book
        """
        return self.num_orders(self.bid)

    def best_bid(self):
        """Get best bid"""
        if self.bid:
            return self.bid[-1]
        return None

    def best_offer(self):
        """Get best offer"""
        if self.offer:
            return self.offer[0]
        return None

    def print(self):
        """
            Prints the order book
        """
        size_bid = len(self.bid)
        size_offer = len(self.offer)
        print("Book[%s]: %d bids, %d offers --> mid @ %f"  % (self.security,
                                                              size_bid, size_offer, self.mid()))
        print("{0: ^32} | {1: ^32}".format("bid", "offer"))
        print("{0:^10},{1:^10},{2:^10} | {3:^10}, {4:^10}, {5:^10}".format(
            "count", "qty", "price", "price", "qty", "count"))

        empty_level = OrderBookLevel("-", "-", "-")
        for i in range(max(size_bid, size_offer)):
            bid = self.bid[-(i+1)] if i < size_bid else empty_level
            offer = self.offer[i] if i < size_offer else empty_level
            print("{0:^10},{1:^10},{2:^10} | {3:^10}, {4:^10}, {5:^10}".format(
                bid.order_count, bid.qty, bid.price, offer.price, offer.qty, offer.order_count))

def gen_orders(config):
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
        qty = random.randint(1, 10)

        yield (mid, qty)

def gen_book(config):
    """Generate order book"""
    print("Using random seed: ", config.randseed)
    random.seed(config.randseed)

    book = OrderBook(security=config.instrument)

    # Initialize book
    for i in range(config.depth):
        delta = (i+1) * 1.0
        book.order(Order(secid=config.instrument,
                         side=Side.BUY, price=(config.base - delta), qty=1))
        book.order(Order(secid=config.instrument,
                         side=Side.SELL, price=(config.base + delta), qty=1))

    bid_series = []
    offer_series = []
    mid_series = []
    for (mid, qty) in gen_orders(config):
        # book.print()
        order = None

        num_offers = book.num_offers()
        num_bids = book.num_bids()
        side = Side.BUY
        # if direction > 0:
        if num_offers > num_bids:
            side = Side.BUY
        elif num_offers < num_bids:
            side = Side.SELL
        else:
            side = Side.SELL if random.randint(0, 1) == 0 else Side.BUY

        order = Order(secid=config.instrument,
                      side=side, price=mid, qty=qty)
        trades = book.order(order)
        if trades:
            print("New trades: ", trades)
        mid_series += [book.mid()]
        bid_series += [book.best_bid().price]
        offer_series += [book.best_offer().price]

    book.print()
    return (bid_series, offer_series, mid_series)

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
                        action="store", type=int, default=2)

    parser.add_argument("-o", "--output", dest="outputfile",
                        help="Output file name",
                        action="store", type=argparse.FileType(mode='w'), default="md.csv")

    return parser.parse_args(argv)

def main(argv):
    """ Market data generator entry point """
    print("Market Data Generator")

    args = parse_args(argv)

    (bid_series, offer_series, mid_series) = gen_book(args)

    pyplot.plot(range(args.samples), numpy.array(mid_series), 'b')
    pyplot.plot(range(args.samples), numpy.array(bid_series), 'g')
    pyplot.plot(range(args.samples), numpy.array(offer_series), 'r')
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv[1:])
