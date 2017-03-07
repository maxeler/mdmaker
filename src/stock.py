#!/usr/bin/python3
"""
Market data generator script.
"""

import argparse
import bisect
import csv
import itertools
import math
import random
import struct
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

MIN_TICK = 0.01
MAX_TICK = 20 * MIN_TICK
DECIMALS = int(abs(math.log10(MIN_TICK)))
DECIMAL_CONVERT = math.pow(10, DECIMALS)
MAX_QUANTITY = 40
VAL_RATIO_MAX = 0.95
SPREAD_MAX = 20 * MIN_TICK
MAX_DEPTH = 20
MAX_QTY = 100

LEVEL_FIELDS = ['price', 'qty', 'order_count']
TRADE_FIELDS = ['price', 'qty', 'aggressor']
FIELDS = ['timestamp', 'secid', 'trade_valid', 'book_valid']


MIN_TIME_DELTA_NANOS = 200
MAX_TIME_DELTA_NANOS = 100000000
def now_nanos():
    """Returns the current simulation time in nanoseconds"""
    if now_nanos.sim_time == 0:
        now_nanos.sim_time += int(time.time() * 1000000000)
    else:
        now_nanos.sim_time += random.randint(MIN_TIME_DELTA_NANOS, MAX_TIME_DELTA_NANOS)

    return now_nanos.sim_time

class Side(Enum):
    """
        Trading Side
    """
    BUY = 1
    SELL = 2

OrderBookLevel = recordclass('OrderBookLevel',
                             LEVEL_FIELDS)
Order = recordclass('Order', ['secid', 'side', 'price', 'qty'])
Trade = recordclass('Trade', TRADE_FIELDS)

LEVEL_FORMAT = "lvl{}_{}_{}"
TRADE_FORMAT = "trade_{}"

class OrderBook(object):
    """An Order Book data model"""
    def __init__(self, security, file):
        self.bid = SortedListWithKey(key=(lambda level: level.price))
        self.offer = SortedListWithKey(key=(lambda level: level.price))
        self.security = security
        self.csv = None
        self.file = file

    def use_csv(self):
        """
            Use a CSV output file rather than binary
        """
        fields = FIELDS + \
                 [LEVEL_FORMAT.format(1, 'bid', field) for field in reversed(LEVEL_FIELDS)] + \
                 [LEVEL_FORMAT.format(1, 'offer', field) for field in LEVEL_FIELDS] + \
                 [TRADE_FORMAT.format(field) for field in TRADE_FIELDS]
        self.csv = csv.DictWriter(f=self.file, fieldnames=fields)
        self.csv.writeheader()

    def cross(self, bid, offer):
        """
            Check if two levels can cross
        """
        if bid.price >= offer.price:
            remaining_qty = min(offer.qty - bid.qty, MAX_QTY)
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


    def compact(self, levels, start):
        """
            Compacts an order list, such that orders at the same price level are merged.
            This assumes the order list is sorted.
        """

        # print("Compacting book")
        # self.print()
        last_level = None
        for i in range(start, -1, -1):
            level = levels[i]
            if last_level:
                if level.price == last_level.price:
                    last_level.qty += level.qty
                    last_level.order_count += level.order_count
                    del levels[i]
                else:
                    break
            else:
                last_level = level

    def order(self, order):
        """
            Apply an order to the Order Book.
            Return a list of trades generated as a result.
        """

        #print("Evaluating order: ", order)
        if self.security != order.secid:
            raise ("Cannot place order for security "
                   "%s on book[%s]" % (order.security, self.security))

        levels = self.bid
        if order.side == Side.SELL:
            levels = self.offer

        new_level = OrderBookLevel(price=order.price, qty=order.qty, order_count=1)
        start_index = levels.bisect_right(new_level)
        levels.insert(start_index, new_level)
        self.compact(levels, start_index)

        # Trim list
        if order.side == Side.SELL:
            # Delete from end of list
            if len(self.offer) > MAX_DEPTH:
                for i in range(len(self.offer) - MAX_DEPTH):
                    del self.offer[-1]
        else:
            # Delete from start of list
            if len(self.bid) > MAX_DEPTH:
                for i in range(len(self.bid) - MAX_DEPTH):
                    del self.bid[0]

        return self.match(order.side)

    def mid(self):
        """
            Get mid price of top of Book
        """
        if self.bid and self.offer:
            return (self.bid[-1].price + self.offer[0].price) / 2.0

        raise Exception("No bids / offers!")

    def spread(self):
        """
            Get the spread at the Top of Book
        """
        if self.bid and self.offer:
            return self.offer[0].price - self.bid[-1].price

        return 0

    def reduce_book_value(self, level1, level2):
        """
            Reduction function for price * qty
        """
        value1 = level1
        value2 = level2
        if isinstance(level1, OrderBookLevel):
            value1 = level1.price * level1.qty
        if isinstance(level2, OrderBookLevel):
            value2 = level2.price * level2.qty
        return value1 + value2

    def book_value(self, levels):
        """
            Computes the value of open orders on a list of book levels
        """
        if len(levels) > 1:
            return reduce(self.reduce_book_value, levels)
        elif len(levels) > 0:
            return levels[0].price * levels[0].qty
        else:
            raise Exception("Cannot reduce empty book")

    def value_offers(self):
        """
            Returns the value of offers on the book
        """
        return self.book_value(self.offer)

    def value_bids(self):
        """
            Returns the value of bids on the book
        """
        return self.book_value(self.bid)

    def depth_offers(self):
        """
            Returns the number of offers on the book
        """
        return len(self.offer)

    def depth_bids(self):
        """
            Returns the number of bids on the book
        """
        return len(self.bid)

    def best_bid(self):
        """Get best bid"""
        if self.bid:
            return self.bid[-1].price
        return None

    def best_offer(self):
        """Get best offer"""
        if self.offer:
            return self.offer[0].price
        return None

    def qty(self, levels, at_level):
        """Get qty at sepcific level"""
        if len(levels) > at_level:
            return levels[at_level].qty
        return 0

    def bid_qty(self, at_level):
        """Get qty at sepcific level"""
        return self.qty(self.bid, at_level)

    def offer_qty(self, at_level):
        """Get qty at sepcific level"""
        return self.qty(self.offer, -at_level)

    def aggregate_bid_qty(self, trade_price):
        """Sum of qty that would match a price"""
        qty = 0
        for i in range(len(self.bid)):
            if self.bid[-i].price >= trade_price:
                qty += self.bid[-i].qty
        return qty

    def aggregate_offer_qty(self, trade_price):
        """Sum of qty that would match a price"""
        qty = 0
        for i in range(len(self.offer)):
            # print("trade_price = {} offer[{}] = {}".format(trade_price, i, self.offer[i].price))
            if self.offer[i].price <= trade_price:
                qty += self.offer[i].qty
                # print("Running qty = {}".format(qty))
        return qty

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

    def csv_book_update(self):
        """
            Add row to CSV file with current book state.
        """
        row = {'timestamp' : now_nanos(),
               'secid' : self.security,
               'trade_valid' : False,
               'book_valid' : 'True'}
        bid = self.bid[-1]._asdict()
        offer = self.offer[0]._asdict()
        row.update({
            LEVEL_FORMAT.format(1, 'bid', f) : bid[f] for f in LEVEL_FIELDS
            })
        row.update({
            LEVEL_FORMAT.format(1, 'offer', f) : offer[f] for f in LEVEL_FIELDS
            })
        row.update({TRADE_FORMAT.format(f) : None for f in TRADE_FIELDS})
        self.csv.writerow(row)

    def bin_book_update(self):
        """
            Adds a binary entry to the file
        """
        trade_update_fmt = "II"
        trade_update_data = [0, 0]
        order_book_level_fmt = "IIIIII"
        levels = [
            (self.bid[-(i+1)].price * DECIMAL_CONVERT,
             self.bid[-(i+1)].qty,
             self.bid[-(i+1)].order_count,
             self.offer[i].price * DECIMAL_CONVERT,
             self.offer[i].qty,
             self.offer[i].order_count) for i in range(5)]
        order_book_level_data = []
        for data in levels:
            order_book_level_data += list(data)
        order_book_level_data = [int(v) for v in order_book_level_data]
        valids_fmt = "I"
        valids_data = [2]
        the_data = [now_nanos(), self.security] + \
                   trade_update_data + order_book_level_data + valids_data
        data = struct.pack("<QI" + trade_update_fmt + order_book_level_fmt * 5 + valids_fmt,
                           *the_data)
        self.file.write(data)

    def csv_trade_update(self, trades):
        """
            Add row to CSV file for the trades
        """
        for trade in trades:
            row = {'timestamp' : now_nanos(),
                   'secid' : self.security,
                   'trade_valid' : True,
                   'book_valid' : 'False'}
            trade_dict = trade._asdict()
            row.update({
                LEVEL_FORMAT.format(1, 'bid', f) : None for f in LEVEL_FIELDS
                })
            row.update({
                LEVEL_FORMAT.format(1, 'offer', f) : None for f in LEVEL_FIELDS
                })
            row.update({TRADE_FORMAT.format(f) : trade_dict[f] for f in TRADE_FIELDS})
            self.csv.writerow(row)

    def bin_trade_update(self, trades):
        """
            Adds a binary entry to the file
        """
        for trade in trades:
            trade_update_fmt = "II"
            trade_update_data = [int(trade.price * DECIMAL_CONVERT), trade.qty]
            order_book_level_fmt = "IIIIII"
            levels = [(0, 0, 0, 0, 0, 0) for i in range(5)]
            order_book_level_data = []
            for data in levels:
                order_book_level_data += list(data)
            order_book_level_data = [int(v) for v in order_book_level_data]
            valids_fmt = "I"
            valids_data = [1]
            the_data = [now_nanos(), self.security] + \
                    trade_update_data + order_book_level_data + valids_data
            data = struct.pack("<QI" + trade_update_fmt + order_book_level_fmt * 5 + valids_fmt,
                               *the_data)
            self.file.write(data)

def gen_orders(config, book):
    """Generator function for the mid price"""
    upper_bound = config.base + config.bound
    lower_bound = config.base - config.bound
    mid = config.base
    direction = 1.0

    for i in range(config.samples):
        if i % config.variation == 0:
            if mid >= upper_bound:
                direction = -1.0
            elif mid <= lower_bound:
                direction = 1.0
            elif random.randint(0, 1) == 0:
                direction = -1.0
            else:
                direction = 1.0

        mid += direction * random.randint(1, 10) * MIN_TICK

        if mid <= lower_bound:
            mid = lower_bound

        if mid >= upper_bound:
            mid = upper_bound

        orders = []
        sell_price = mid + MIN_TICK
        buy_price = mid - MIN_TICK
        if direction < 0:
            qty = book.aggregate_bid_qty(sell_price)
            orders += [Order(secid=config.instrument,
                             side=Side.SELL,
                             price=round(sell_price, DECIMALS),
                             qty=qty + 1)]
            orders += [Order(secid=config.instrument,
                             side=Side.BUY,
                             price=round(buy_price, DECIMALS),
                             qty=1)]
        else:
            qty = book.aggregate_offer_qty(buy_price)
            orders += [Order(secid=config.instrument,
                             side=Side.BUY,
                             price=round(buy_price, DECIMALS),
                             qty=qty+1)]
            orders += [Order(secid=config.instrument,
                             side=Side.SELL,
                             price=round(sell_price, DECIMALS),
                             qty=1)]

        orders += pad_book(config, book.depth_bids(), buy_price, Side.BUY)
        orders += pad_book(config, book.depth_offers(), sell_price, Side.SELL)

        yield (orders, mid)

def pad_book(config, depth, price, side):
    """
        Add orders to the book such that we are at full depths.
    """
    orders = []
    sign = -1.0 if side == Side.BUY else 1.0
    if depth < MAX_DEPTH:
        num = MAX_DEPTH - depth
        best = price
        offset = 1
        for _ in itertools.repeat(None, num):
            orders += [Order(secid=config.instrument,
                             side=side,
                             price=round(best + sign * offset * MIN_TICK, DECIMALS),
                             qty=random.randint(1, 10))]
            offset += random.randint(1, 3)
    return orders

def gen_book(config):
    """Generate order book"""
    print("Using random seed: ", config.randseed)
    random.seed(config.randseed)

    book = OrderBook(security=config.instrument, file=config.outputfile)

    if config.csv:
        book.use_csv()

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
    book_mid_series = []
    for (orders, mid) in gen_orders(config, book):
        # book.print()
        # print("Target mid --> {}".format(mid))
        for order in orders:
            # print("New order: ", order)
            trades = book.order(order)
            if config.csv:
                book.csv_trade_update(trades)
            else:
                book.bin_trade_update(trades)

        if config.csv:
            book.csv_book_update()
        else:
            book.bin_book_update()

        mid_series += [mid]
        book_mid_series += [book.mid()]
        bid_series += [book.best_bid()]
        offer_series += [book.best_offer()]

    book.print()
    return (bid_series, offer_series, mid_series, book_mid_series)

def parse_args(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", dest="samples",
                        help="Number of samples required",
                        action="store", type=int, default=1000)

    parser.add_argument("-b", "--base", dest="base",
                        help="Starting price",
                        action="store", type=float, default=200.0)

    parser.add_argument("-c", "--csv", dest="csv",
                        help="Use CSV format [default Binary]",
                        action="store_true", default=False)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Book depth",
                        action="store", type=int, default=5)

    parser.add_argument("-i", "--instrument", dest="instrument",
                        help="Security ID",
                        action="store", type=int, default=123456)

    parser.add_argument("-v", "--variation", dest="variation",
                        help="Change in direction - lower is more often",
                        action="store", type=int, default=4)

    parser.add_argument("-u", "--bound", dest="bound",
                        help="Price bound",
                        action="store", type=float, default=100.0)

    parser.add_argument("-g", "--debug", dest="debug",
                        help="Enable debugging",
                        action="store_true", default=False)

    parser.add_argument("-r", "--randomseed", dest="randseed",
                        help="Random seed",
                        action="store", type=int, default=6)

    parser.add_argument("-o", "--output", dest="outputfile",
                        help="Output file name",
                        action="store", type=argparse.FileType(mode='bw'), default="md")

    return parser.parse_args(argv)

def main(argv):
    """ Market data generator entry point """
    print("Market Data Generator")

    now_nanos.sim_time = 0
    args = parse_args(argv)
    print("Output file '{}' in {} format".format(args.outputfile.name,
                                                 'CSV' if args.csv else 'binary'))

    if args.csv:
        name = args.outputfile.name
        args.outputfile.close()
        args.outputfile = open(name, "w")

    (bid_series, offer_series, mid_series, book_mid_series) = gen_book(args)

    # pyplot.plot(range(args.samples), numpy.array(mid_series), 'blue')
    pyplot.plot(range(args.samples), numpy.array(bid_series), 'green')
    pyplot.plot(range(args.samples), numpy.array(offer_series), 'red')
    # pyplot.plot(range(args.samples), numpy.array(book_mid_series), 'black')
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv[1:])
