import { Document } from '@langchain/core/documents';
import { color, colorize } from 'json-colorizer';

export type Documents = Document[];

export function prettyPrint(obj: any) {

  const defaultTheme = {
    Whitespace: color.gray,
    Brace: color.gray,
    Bracket: color.gray,
    Colon: color.gray,
    Comma: color.gray,
    StringKey: color.magenta,
    StringLiteral: color.yellow,
    NumberLiteral: color.green,
    BooleanLiteral: color.cyan,
    NullLiteral: color.white
};

  console.log(colorize(
    obj,
    {
      colors: {
        ...defaultTheme,
        StringKey: color.green,
        StringLiteral: color.white,
      }
    }
  ));
}