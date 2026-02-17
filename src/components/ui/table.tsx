import * as React from "react";

import { cn } from "@/lib/utils";

const Table = React.forwardRef<
  React.ElementRef<"table">,
  React.ComponentPropsWithoutRef<"table">
>(({ className, ...props }, ref) => (
  <div className="w-full overflow-auto">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
));
Table.displayName = "Table";

const TableHead = React.forwardRef<
  React.ElementRef<"thead">,
  React.ComponentPropsWithoutRef<"thead">
>(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("[&_tr]:border-b", className)} {...props} />
));
TableHead.displayName = "TableHead";

const TableBody = React.forwardRef<
  React.ElementRef<"tbody">,
  React.ComponentPropsWithoutRef<"tbody">
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0", className)}
    {...props}
  />
));
TableBody.displayName = "TableBody";

const TableRow = React.forwardRef<
  React.ElementRef<"tr">,
  React.ComponentPropsWithoutRef<"tr">
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b border-[var(--border-strong)]/50 transition-colors",
      className
    )}
    {...props}
  />
));
TableRow.displayName = "TableRow";

const TableCell = React.forwardRef<
  React.ElementRef<"td">,
  React.ComponentPropsWithoutRef<"td">
>(({ className, ...props }, ref) => (
  <td ref={ref} className={cn("p-2 align-middle", className)} {...props} />
));
TableCell.displayName = "TableCell";

export { Table, TableBody, TableCell, TableHead, TableRow };
